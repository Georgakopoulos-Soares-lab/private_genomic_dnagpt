# -*- coding: utf-8 -*-
"""Toy DNAGPT transformer block under **Scheme B**: hybrid client-assisted CKKS.

Companion/contrast to ``toy_block_ckks.py`` (Scheme A, frozen baseline). Same toy
config (D=8, T=4, 2 heads) and the same plaintext oracle (``oracle.block_forward``),
so results are directly comparable. See
``docs/shared/architecture_options.md`` for the full architecture rationale.

Protocol change from Scheme A: linear algebra (LayerNorm centering/variance, QKV/
projection/MLP matmuls, residual adds) stays in one uninterrupted CKKS ciphertext
lineage on the "server" role, which never receives the secret key. At each
nonlinearity, the **minimum** information needed crosses to the "client" role (the
data owner, who already holds the secret key and only ever decrypts ciphertexts
derived from its own query):

  - LayerNorm: only the scalar `variance` (already includes eps) per token crosses.
    The client computes the exact `1/sqrt(variance)` and re-encrypts it; the LN
    scale/bias (server IP) are applied afterwards, still encrypted.
  - Causal softmax: the packed causal attention scores for one token (all heads,
    one slot per (head, position) pair) cross. The client computes the exact
    softmax per head and re-encrypts the packed weight vector; V stays encrypted
    and is scaled/summed server-side.
  - GELU: the full MLP hidden pre-activation (4D per token) crosses, because GELU is
    a genuine elementwise nonlinearity with no smaller sufficient statistic. This is
    the widest boundary and is reported as such.

No Chebyshev polynomial approximation and no bootstrap are needed anywhere, because
depth is never allowed to accumulate across a boundary — every re-encrypted value is
a fresh, level-0 ciphertext. Measured on this toy config, the deepest point in the
whole block (the ln2 -> gelu segment) reaches level 14 -- far below Scheme A's 49,
and comfortably inside OpenFHE's 65536-ring-dimension bucket (depth 12-22 at
HEStd_128_classic/ScalingModSize=50 on this host), vs. Scheme A's 131072. This ring
dimension gap, not just the depth number, is where Scheme B's real per-operation
speedup comes from.

Run (from repo root):
  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work dnagpt-openfhe \
      python3 fhe/block/toy_block_ckks_scheme_b.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oracle import block_forward, gate, toy_weights  # noqa: E402

D, T, N_HEADS = 8, 4, 2
HEAD_DIM = D // N_HEADS
MLP_DIM = 4 * D
BATCH_SLOTS = MLP_DIM  # widest value that ever needs packing (the GELU boundary)
EPS = 1e-5
# Depth budget: every value crossing a boundary re-enters at level 0, so depth only
# has to cover the *shallow* segment between two boundaries (mean/variance, one
# matmul, one score dot-product), not the whole block. sum_bcast's correctness fix
# (isolate slot 0 + doubling-replicate) adds one extra mult level per broadcast
# call versus the naive (buggy) version, so this has more margin than the
# level-14 measured for the buggy variant. DEPTH=20 keeps us in OpenFHE's
# 65536-ring-dimension bucket (empirically depth 12-22 on this host,
# HEStd_128_classic/ScalingModSize=50) rather than the far more expensive 131072
# bucket that starts at depth 24 -- that ring-dimension gap, not the depth number
# itself, is where Scheme B's real per-operation speedup over Scheme A comes from.
DEPTH = 20
SCALE = 50


# ---------- CKKS context + helpers -------------------------------------------
def build_context():
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        SecurityLevel,
    )

    p = CCParamsCKKSRNS()
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetMultiplicativeDepth(DEPTH)
    p.SetScalingModSize(SCALE)
    p.SetBatchSize(BATCH_SLOTS)
    cc = GenCryptoContext(p)
    for f in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.KEYSWITCH,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
    ):
        cc.Enable(f)
    return cc


class Client:
    """Holds the secret key. Decrypts only ciphertexts derived from its own query,
    at pre-declared nonlinearity boundaries, and re-encrypts fresh (level-0)
    ciphertexts. Every call is counted as one round trip."""

    def __init__(self, cc, keys):
        self._cc = cc
        self._keys = keys
        self.round_trips = 0
        self.boundary_log = []

    def pt(self, vec):
        return self._cc.MakeCKKSPackedPlaintext(list(vec))

    def encrypt(self, vec):
        return self._cc.Encrypt(self._keys.publicKey, self.pt(vec))

    def _decrypt(self, ct, n):
        # Pinned OpenFHE 1.5.1 binding argument order (matches toy_block_ckks.py).
        plaintext = self._cc.Decrypt(ct, self._keys.secretKey)
        plaintext.SetLength(n)
        return np.array(plaintext.GetRealPackedValue())

    def cross_boundary(self, name, ct, n, transform):
        """Decrypt n values, apply the exact plaintext nonlinearity, re-encrypt.

        Counts one round trip and logs how many real values crossed, so the
        evidence record can state the exact boundary width, not just its name.
        """
        values = self._decrypt(ct, n)
        result = transform(values)
        self.round_trips += 1
        self.boundary_log.append({"name": name, "values_crossed": int(n)})
        return self.encrypt(result)

    def final_readout(self, ct, n):
        """The harness's own final measurement decrypt -- not a protocol round trip,
        matches Scheme A's single end-of-lineage decrypt for oracle comparison."""
        return self._decrypt(ct, n)


class EvalOnlyContext:
    """Proxy enforcing the server role never touches the secret key or decrypts."""

    def __init__(self, cc):
        self._cc = cc
        self.decrypt_attempts = 0

    def __getattr__(self, name):
        if name == "Decrypt":
            self.decrypt_attempts += 1
            raise AssertionError("Decrypt is forbidden on the server role")
        return getattr(self._cc, name)


def onehot(cc, j, width=BATCH_SLOTS):
    v = [0.0] * width
    v[j] = 1.0
    return cc.MakeCKKSPackedPlaintext(v)


def sum_bcast(cc, ct, n):
    """Sum the first n (real, zero-padded-beyond) slots and broadcast the TRUE
    total into every slot of the full BATCH_SLOTS window.

    OpenFHE's EvalSum(ct, n) is a *sliding* cyclic-window sum, not a per-block
    broadcast: it only equals the true sum of the first n slots at slot 0 (every
    other slot mixes in wraparound/zero contributions from outside [0, n)). That
    windowed-broadcast assumption is only valid when the input is already
    n-periodic across the whole batch (the trick Scheme A's toy_block_ckks.py
    uses deliberately -- see its ``pt()`` docstring). Scheme B's boundary values
    are not replicated that way, so we isolate the one correct value (slot 0)
    with a one-hot mask and then explicitly replicate it across BATCH_SLOTS via
    right-rotation doubling (needs the negative power-of-two rotation keys
    generated once in ``main()``).
    """
    windowed = cc.EvalSum(ct, n)
    out = cc.EvalMult(windowed, onehot(cc, 0))
    shift = 1
    while shift < BATCH_SLOTS:
        out = cc.EvalAdd(out, cc.EvalRotate(out, -shift))
        shift *= 2
    return out


def dense_linear(cc, ct, w, b, n_in):
    """y = W @ x + b via one ct*pt mult + one select mult per output row.

    Not depth-optimized (no BSGS diagonal packing) -- Scheme B does not need it,
    since no lineage segment survives more than a couple of levels before its next
    boundary. This generalizes past the square-matrix-only diagonal method to the
    non-square QKV/MLP projections without extra bookkeeping.
    """
    n_out = w.shape[0]
    out = None
    for row in range(n_out):
        weighted = cc.EvalMult(ct, cc.MakeCKKSPackedPlaintext(list(w[row])))
        summed = sum_bcast(cc, weighted, n_in)
        select = [0.0] * row + [1.0]
        term = cc.EvalMult(summed, cc.MakeCKKSPackedPlaintext(select))
        out = term if out is None else cc.EvalAdd(out, term)
    return cc.EvalAdd(out, cc.MakeCKKSPackedPlaintext(list(b)))


def layernorm_scheme_b(cc, client, ct, w, b, boundary_name):
    n = len(w)
    mu = cc.EvalMult(sum_bcast(cc, ct, n), cc.MakeCKKSPackedPlaintext([1.0 / n] * n))
    centered = cc.EvalSub(ct, mu)
    variance = cc.EvalMult(
        sum_bcast(cc, cc.EvalMult(centered, centered), n),
        cc.MakeCKKSPackedPlaintext([1.0 / n] * n),
    )
    variance = cc.EvalAdd(variance, cc.MakeCKKSPackedPlaintext([EPS] * n))

    def invstd_exact(values):
        return [1.0 / math.sqrt(values[0])] * n

    invstd_ct = client.cross_boundary(boundary_name, variance, 1, invstd_exact)
    out = cc.EvalMult(centered, invstd_ct)
    out = cc.EvalMult(out, cc.MakeCKKSPackedPlaintext(list(w)))
    return cc.EvalAdd(out, cc.MakeCKKSPackedPlaintext(list(b)))


def attention_scheme_b(cc, client, ln_cts, p, n_heads):
    T_local = len(ln_cts)
    head_dim = D // n_heads
    scale = 1.0 / math.sqrt(head_dim)
    Wq, Wk, Wv = p["attn_w"][:D], p["attn_w"][D : 2 * D], p["attn_w"][2 * D :]
    bq, bk, bv = p["attn_b"][:D], p["attn_b"][D : 2 * D], p["attn_b"][2 * D :]
    q = [dense_linear(cc, h, Wq, bq, D) for h in ln_cts]
    k = [dense_linear(cc, h, Wk, bk, D) for h in ln_cts]
    v = [dense_linear(cc, h, Wv, bv, D) for h in ln_cts]

    head_masks = [
        [1.0 if head * head_dim <= d < (head + 1) * head_dim else 0.0 for d in range(D)]
        for head in range(n_heads)
    ]
    v_heads = [
        [cc.EvalMult(value, cc.MakeCKKSPackedPlaintext(head_masks[h])) for value in v]
        for h in range(n_heads)
    ]

    attn_out = []
    for i in range(T_local):
        # Pack every head's causal scores for token i into distinct slots so one
        # decrypt/re-encrypt pair covers the whole token, not one call per head.
        packed_scores = None
        for h in range(n_heads):
            mask = head_masks[h]
            for j in range(i + 1):
                dot = sum_bcast(
                    cc,
                    cc.EvalMult(
                        cc.EvalMult(q[i], k[j]), cc.MakeCKKSPackedPlaintext(mask)
                    ),
                    D,
                )
                slot = h * T_local + j
                select = [0.0] * slot + [1.0]
                term = cc.EvalMult(dot, cc.MakeCKKSPackedPlaintext(select))
                packed_scores = (
                    term if packed_scores is None else cc.EvalAdd(packed_scores, term)
                )

        def softmax_exact(values, i=i, n_heads=n_heads, T_local=T_local, scale=scale):
            out = [0.0] * (n_heads * T_local)
            for h in range(n_heads):
                row = [values[h * T_local + j] * scale for j in range(i + 1)]
                m = max(row)
                exps = [math.exp(s - m) for s in row]
                z = sum(exps)
                for j in range(i + 1):
                    out[h * T_local + j] = exps[j] / z
            return out

        weights_ct = client.cross_boundary(
            f"attention_softmax_token{i}",
            packed_scores,
            n_heads * T_local,
            softmax_exact,
        )

        out_i = None
        for h in range(n_heads):
            head_out = None
            for j in range(i + 1):
                slot = h * T_local + j
                select_slot = [0.0] * slot + [1.0]
                weight_bcast = sum_bcast(
                    cc,
                    cc.EvalMult(weights_ct, cc.MakeCKKSPackedPlaintext(select_slot)),
                    n_heads * T_local,
                )
                term = cc.EvalMult(weight_bcast, v_heads[h][j])
                head_out = term if head_out is None else cc.EvalAdd(head_out, term)
            out_i = head_out if out_i is None else cc.EvalAdd(out_i, head_out)
        attn_out.append(dense_linear(cc, out_i, p["proj_w"], p["proj_b"], D))
    return attn_out


def gelu_scheme_b(cc, client, ln_ct, p, token_index):
    hidden = dense_linear(cc, ln_ct, p["fc_w"], p["fc_b"], D)

    def gelu_exact(values):
        arr = np.asarray(values[:MLP_DIM])
        return list(
            0.5
            * arr
            * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (arr + 0.044715 * arr**3)))
        )

    activated = client.cross_boundary(
        f"gelu_token{token_index}", hidden, MLP_DIM, gelu_exact
    )
    return dense_linear(cc, activated, p["fcp_w"], p["fcp_b"], MLP_DIM)


def encrypted_block_scheme_b(cc, client, x_cts, p, n_heads):
    """Evaluate the block. ``cc`` here is the server's key-less proxy; ``client``
    is the only object holding the secret key, invoked solely at declared
    boundaries. Linear algebra stays in one lineage between boundaries."""
    trace = {}

    def record(name, cts):
        levels = [int(ct.GetLevel()) for ct in cts]
        trace[name] = levels
        print(f"{name}: level min={min(levels)} max={max(levels)}", flush=True)

    ln1 = [
        layernorm_scheme_b(cc, client, ct, p["ln1_w"], p["ln1_b"], f"ln1_token{i}")
        for i, ct in enumerate(x_cts)
    ]
    record("ln1", ln1)

    attn = attention_scheme_b(cc, client, ln1, p, n_heads)
    record("attention_projection", attn)

    residual1 = [cc.EvalAdd(x_cts[i], attn[i]) for i in range(len(x_cts))]
    record("residual1", residual1)

    ln2 = [
        layernorm_scheme_b(cc, client, ct, p["ln2_w"], p["ln2_b"], f"ln2_token{i}")
        for i, ct in enumerate(residual1)
    ]
    record("ln2", ln2)

    mlp = [gelu_scheme_b(cc, client, ln2[i], p, i) for i in range(len(ln2))]
    record("mlp_projection", mlp)

    outputs = [cc.EvalAdd(residual1[i], mlp[i]) for i in range(len(residual1))]
    record("block_output", outputs)
    return outputs, trace


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="write immutable results/runs/<tag>.json")
    parser.add_argument(
        "--environment",
        default="Docker linux/amd64 on Apple Silicon [emu]",
    )
    parser.add_argument(
        "--latency-label",
        choices=("[emu]", "[native-cpu]"),
        default="[emu]",
    )
    parser.add_argument("--container-image", default="dnagpt-openfhe")
    args = parser.parse_args()
    try:
        import openfhe  # noqa: F401
    except Exception:
        print("[skip] openfhe absent — run in the dnagpt-openfhe Docker image.")
        return 0

    x, p = toy_weights(D=D, T=T, n_heads=N_HEADS)
    oracle = block_forward(x, p, n_heads=N_HEADS)

    assert D % N_HEADS == 0
    setup_started = time.perf_counter()
    cc = build_context()
    ring = cc.GetRingDimension()
    print(
        f"OpenFHE CKKS context (Scheme B): ring_dim={ring}, mult_depth={DEPTH}, "
        f"128-bit; D={D} T={T} heads={N_HEADS}",
        flush=True,
    )
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    # Negative power-of-two rotation keys for sum_bcast's right-rotation
    # doubling-replicate (see its docstring) -- EvalSumKeyGen only generates the
    # positive-direction keys its own sliding-window sum needs.
    neg_rotations = []
    shift = 1
    while shift < BATCH_SLOTS:
        neg_rotations.append(-shift)
        shift *= 2
    cc.EvalRotateKeyGen(keys.secretKey, neg_rotations)
    setup_keygen_s = time.perf_counter() - setup_started
    print(f"context + keygen: {setup_keygen_s:.1f}s", flush=True)

    client = Client(cc, keys)
    x_cts = [client.encrypt(x[i]) for i in range(T)]

    t0 = time.perf_counter()
    eval_cc = EvalOnlyContext(cc)
    out_cts, level_trace = encrypted_block_scheme_b(eval_cc, client, x_cts, p, N_HEADS)
    assert eval_cc.decrypt_attempts == 0
    latency = time.perf_counter() - t0

    got = np.array([client.final_readout(ct, D) for ct in out_cts])
    levels_before_readout = [int(ct.GetLevel()) for ct in out_cts]

    g = gate(got, oracle)
    boundary_values_crossed = sum(
        entry["values_crossed"] for entry in client.boundary_log
    )
    result = {
        "schema_version": 1,
        "scheme": "B",
        "task": "FHE toy DNAGPT block (CKKS/OpenFHE), hybrid client-assisted",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "OpenFHE CKKS",
        "security": "HEStd_128_classic",
        "environment": args.environment,
        "provenance": {
            "script_sha256": file_sha256(__file__),
            "oracle_sha256": file_sha256(
                os.path.join(os.path.dirname(__file__), "..", "oracle.py")
            ),
            "container_image": args.container_image,
            "openfhe_version": importlib.metadata.version("openfhe"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "config": {
            "D": D,
            "T": T,
            "heads": N_HEADS,
            "ring_dim": ring,
            "batch_slots": BATCH_SLOTS,
            "mult_depth": DEPTH,
            "levels_before_readout": levels_before_readout,
            "level_trace": level_trace,
            "bootstraps": 0,
            "chebyshev_calls": 0,
        },
        "protocol": {
            "round_trips": client.round_trips,
            "boundary_log": client.boundary_log,
            "total_values_crossed_to_client": int(boundary_values_crossed),
            "note": (
                "Client is the data owner and already holds the secret key; it "
                "decrypts only ciphertexts derived from its own query, at these "
                "pre-declared boundaries, and re-encrypts. The server role "
                "(EvalOnlyContext) never receives the secret key and made zero "
                "decrypt attempts (asserted)."
            ),
        },
        "global_rel_inf": g["global_rel_inf"],
        "worst_token_rel_inf": g["worst_token_rel_inf"],
        "tol": g["tol"],
        "passed": g["passed"],
        "context_setup_keygen_s": setup_keygen_s,
        "evaluation_latency_s": latency,
        "latency_label": args.latency_label,
        "server_decrypt_attempts": eval_cc.decrypt_attempts,
        "final_readout_decrypt_calls": len(out_cts),
    }
    print(json.dumps(result, indent=2))
    print("oracle[0]:", np.round(oracle[0], 4))
    print("fhe   [0]:", np.round(got[0], 4))

    if args.tag:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.tag}.json")
        if os.path.exists(out_path):
            raise FileExistsError(f"refusing to overwrite immutable run: {out_path}")
        with open(out_path, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"[evidence] {out_path}")
    print(
        f"\n[{'pass' if g['passed'] else 'FAIL'}] "
        f"encrypted block rel_inf={g['global_rel_inf']:.2e} (tol {g['tol']}), "
        f"{client.round_trips} client round trips, max depth "
        f"{max(max(v) for v in level_trace.values())}"
    )
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
