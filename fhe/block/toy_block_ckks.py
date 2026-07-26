# -*- coding: utf-8 -*-
"""Toy DNAGPT transformer block, fully encrypted end-to-end under CKKS (OpenFHE).

THE existence proof for Phase B: encrypt the (embedded) input tokens, run a complete
DNAGPT-shaped block — LayerNorm, QKV, causal softmax attention, projection, LayerNorm,
GELU MLP, residuals — entirely on ciphertexts in ONE CryptoContext lineage, and decrypt
EXACTLY ONCE at the very end to compare against the plaintext oracle. No decryption
occurs at any intermediate step (no client round-trips, unlike Concrete ML's hybrid LLM).

Layout: per-token ciphertexts with each D-slot token repeated across T blocks. Tiny
config (D=8, T=4, 2 heads) keeps it runnable under linux/amd64 emulation on a Mac while
exercising multi-head causal attention. The T encrypted outputs are masked into one
packed ciphertext and decrypted with one final Decrypt call. Latency here is [emu], not
representative — see docs/pure/.

Run (from repo root):
  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work dnagpt-openfhe \
      python3 fhe/block/toy_block_ckks.py
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
BATCH_SLOTS = D * T
# Depth 46 exhausted at the first GELU in the initial two-head run. Depth 50
# crosses OpenFHE's next ring-dimension boundary (131072 -> 262144), so the
# circuit uses two depth-neutral algebraic folds below and the largest secure
# depth that retains ring 131072.
DEPTH = 49
SCALE = 50
DEG_INVSQRT = 27  # 1/sqrt(var+eps), steep near 0 -> higher degree
DEG_EXP = 13
DEG_RECIP = 27
DEG_GELU = 7
EPS = 1e-5
INVSQRT_LO, INVSQRT_HI = 0.03, 0.9  # var+eps domain
EXP_LO, EXP_HI = -2.0, 2.0  # attention logits domain
RECIP_LO, RECIP_HI = 0.5, 4.5  # softmax-sum domain (row0 sum~1 .. row3 sum~3.6)
# The deterministic toy contract's plaintext FC values are [-1.381, 1.297].
# This source-frozen band leaves explicit headroom around the toy activation range.
GELU_LO, GELU_HI = -2.0, 2.0


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


def dec(cc, keys, ct, n=BATCH_SLOTS):
    # The pinned OpenFHE 1.5.1 binding uses this argument order. Do not add a
    # retry fallback: the acceptance contract requires exactly one final call.
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(n)
    return np.array(pt.GetRealPackedValue())


def raw_pt(cc, vec):
    vec = list(vec)
    if len(vec) != BATCH_SLOTS:
        raise ValueError(f"expected {BATCH_SLOTS} slots, got {len(vec)}")
    return cc.MakeCKKSPackedPlaintext(vec)


def pt(cc, vec):
    """Encode one D-slot value replicated across T blocks.

    Repetition makes D-wide rotations and EvalSum behave identically in every block.
    The final pack masks a different block from each output ciphertext.
    """
    vec = list(vec)
    if len(vec) != D:
        raise ValueError(f"expected {D} values, got {len(vec)}")
    return raw_pt(cc, vec * T)


def block_pt(cc, vec, token):
    """Encode one D-slot vector only in token's final output block."""
    vec = list(vec)
    if len(vec) != D:
        raise ValueError(f"expected {D} values, got {len(vec)}")
    slots = [0.0] * BATCH_SLOTS
    start = token * D
    slots[start : start + D] = vec
    return raw_pt(cc, slots)


def enc(cc, keys, vec):
    return cc.Encrypt(keys.publicKey, pt(cc, vec))


def sum_bcast(cc, ct):  # sum all D slots -> broadcast
    return cc.EvalSum(ct, D)


def mean_bcast(cc, ct):
    return cc.EvalMult(sum_bcast(cc, ct), pt(cc, [1.0 / D] * D))


def matmul(cc, ct, W, b):  # y = W·x + b, W square [D,D]
    n = W.shape[0]
    diags = [[W[i, (i + d) % n] for i in range(n)] for d in range(n)]
    n1 = math.isqrt(n)
    while n % n1:
        n1 += 1
    n2 = n // n1
    baby = {0: ct}
    for j in range(1, n1):
        baby[j] = cc.EvalRotate(ct, j)
    acc = None
    for k in range(n2):
        inner = None
        for j in range(n1):
            d = n1 * k + j
            diag = np.roll(diags[d], n1 * k)
            term = cc.EvalMult(baby[j], pt(cc, diag))
            inner = term if inner is None else cc.EvalAdd(inner, term)
        if k:
            inner = cc.EvalRotate(inner, n1 * k)
        acc = inner if acc is None else cc.EvalAdd(acc, inner)
    return cc.EvalAdd(acc, pt(cc, list(b)))


def matmul_to_block(cc, ct, W, b, token):
    """Diagonal matmul whose existing plaintext products also pack its output.

    The input is repeated in every D-slot block. Encoding each diagonal only in
    ``token``'s destination block makes the result sparse without a subsequent
    ciphertext-plaintext mask multiplication, which would exceed the level-49 chain.
    """
    n = W.shape[0]
    diags = [[W[i, (i + d) % n] for i in range(n)] for d in range(n)]
    acc = None
    for d, diag in enumerate(diags):
        rotated = ct if d == 0 else cc.EvalRotate(ct, d)
        term = cc.EvalMult(rotated, block_pt(cc, diag, token))
        acc = term if acc is None else cc.EvalAdd(acc, term)
    return cc.EvalAdd(acc, block_pt(cc, list(b), token))


def layernorm(cc, ct, w, b):
    mu = mean_bcast(cc, ct)
    cen = cc.EvalSub(ct, mu)
    var = mean_bcast(cc, cc.EvalMult(cen, cen))
    var = cc.EvalAdd(var, pt(cc, [EPS] * D))
    inv = cc.EvalChebyshevFunction(
        lambda t: 1.0 / math.sqrt(t), var, INVSQRT_LO, INVSQRT_HI, DEG_INVSQRT
    )
    out = cc.EvalMult(cen, inv)
    out = cc.EvalMult(out, pt(cc, list(w)))
    return cc.EvalAdd(out, pt(cc, list(b)))


def onehot(cc, j):
    v = [0.0] * D
    v[j] = 1.0
    return pt(cc, v)


def dot_bcast(cc, a, bt):  # <a,b> broadcast to all slots
    return sum_bcast(cc, cc.EvalMult(a, bt))


def select_bcast(cc, ct, j):  # value at slot j -> broadcast
    return sum_bcast(cc, cc.EvalMult(ct, onehot(cc, j)))


# ---------- the encrypted block ----------------------------------------------
def encrypted_block(cc, x_cts, p):
    """Evaluate the block without access to a secret key or Decrypt."""
    trace = {}

    def record_stage(name, ciphertexts):
        stage_levels = [int(ciphertext.GetLevel()) for ciphertext in ciphertexts]
        trace[name] = stage_levels
        print(
            f"{name}: level min={min(stage_levels)} max={max(stage_levels)}",
            flush=True,
        )

    head_dim = D // N_HEADS
    scale = 1.0 / math.sqrt(head_dim)
    # --- attn(ln_1(x)) ---
    ln = [layernorm(cc, ct, p["ln1_w"], p["ln1_b"]) for ct in x_cts]
    record_stage("ln1", ln)
    Wq, Wk, Wv = p["attn_w"][:D], p["attn_w"][D : 2 * D], p["attn_w"][2 * D :]
    bq, bk, bv = p["attn_b"][:D], p["attn_b"][D : 2 * D], p["attn_b"][2 * D :]
    q = [matmul(cc, h, Wq, bq) for h in ln]
    k = [matmul(cc, h, Wk, bk) for h in ln]
    v = [matmul(cc, h, Wv, bv) for h in ln]
    record_stage("qkv", q + k + v)
    head_masks = [
        [1.0 if head * head_dim <= d < (head + 1) * head_dim else 0.0 for d in range(D)]
        for head in range(N_HEADS)
    ]
    # Mask V at its shallow level. This is algebraically identical to masking
    # each completed head output, but removes one level from the critical path.
    v_heads = [
        [cc.EvalMult(value, pt(cc, head_masks[head])) for value in v]
        for head in range(N_HEADS)
    ]

    attn = []
    for i in range(T):
        out_i = None
        for head in range(N_HEADS):
            head_mask = head_masks[head]

            def head_dot(a, b):
                products = cc.EvalMult(cc.EvalMult(a, b), pt(cc, head_mask))
                return sum_bcast(cc, products)

            # Causal scores occupy slots 0..i. Each head gets its own softmax.
            row = None
            for j in range(i + 1):
                score_slot = [0.0] * D
                score_slot[j] = scale
                term = cc.EvalMult(head_dot(q[i], k[j]), pt(cc, score_slot))
                row = term if row is None else cc.EvalAdd(row, term)
            e = cc.EvalChebyshevFunction(math.exp, row, EXP_LO, EXP_HI, DEG_EXP)
            causal_mask = [1.0 if j <= i else 0.0 for j in range(D)]
            e = cc.EvalMult(e, pt(cc, causal_mask))  # zero j>i
            ssum = sum_bcast(cc, e)
            inv = cc.EvalDivide(ssum, RECIP_LO, RECIP_HI, DEG_RECIP)
            soft = cc.EvalMult(e, inv)  # weights in slots 0..i
            head_out = cc.EvalMult(select_bcast(cc, soft, 0), v_heads[head][0])
            for j in range(1, i + 1):
                head_out = cc.EvalAdd(
                    head_out,
                    cc.EvalMult(select_bcast(cc, soft, j), v_heads[head][j]),
                )
            out_i = head_out if out_i is None else cc.EvalAdd(out_i, head_out)
        attn.append(matmul(cc, out_i, p["proj_w"], p["proj_b"]))
        print(f"attention token {i + 1}/{T}: level={attn[-1].GetLevel()}", flush=True)
    record_stage("attention_projection", attn)
    x_cts = [cc.EvalAdd(x_cts[i], attn[i]) for i in range(T)]
    record_stage("residual1", x_cts)

    # --- mlp(ln_2(x)) : fc D->4D (4 chunks), GELU, fcp 4D->D ---
    ln = [layernorm(cc, ct, p["ln2_w"], p["ln2_b"]) for ct in x_cts]
    record_stage("ln2", ln)
    Fc, Fcp = p["fc_w"], p["fcp_w"]  # [4D,D], [D,4D]
    mlp = []
    for i in range(T):
        chunks = []
        for c in range(4):
            Wc = Fc[c * D : (c + 1) * D]  # [D,D]
            bc = p["fc_b"][c * D : (c + 1) * D]
            hc = matmul(cc, ln[i], Wc, bc)
            hc = cc.EvalChebyshevFunction(
                lambda t: 0.5
                * t
                * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (t + 0.044715 * t**3))),
                hc,
                GELU_LO,
                GELU_HI,
                DEG_GELU,
            )
            chunks.append(hc)
        # Fold the final output-block mask into these diagonal plaintexts. The
        # final token ciphertexts are therefore already disjoint and can be
        # packed by depth-neutral additions.
        out = matmul_to_block(cc, chunks[0], Fcp[:, 0:D], p["fcp_b"], i)
        for c in range(1, 4):
            out = cc.EvalAdd(
                out,
                matmul_to_block(
                    cc,
                    chunks[c],
                    Fcp[:, c * D : (c + 1) * D],
                    np.zeros(D),
                    i,
                ),
            )
        mlp.append(out)
        print(f"mlp token {i + 1}/{T}: level={out.GetLevel()}", flush=True)
    residuals = [cc.EvalMult(x_cts[i], block_pt(cc, [1.0] * D, i)) for i in range(T)]
    outputs = [cc.EvalAdd(residuals[i], mlp[i]) for i in range(T)]
    record_stage("block_output", outputs)
    return outputs, trace


class EvalOnlyContext:
    """Proxy that fails closed if encrypted evaluation attempts to decrypt."""

    def __init__(self, cc):
        self._cc = cc
        self.decrypt_attempts = 0

    def __getattr__(self, name):
        if name == "Decrypt":
            self.decrypt_attempts += 1
            raise AssertionError("Decrypt is forbidden during encrypted evaluation")
        return getattr(self._cc, name)


def pack_outputs(cc, out_cts):
    """Add already-disjoint token blocks into one [T,D] ciphertext."""
    packed = out_cts[0]
    for ct in out_cts[1:]:
        packed = cc.EvalAdd(packed, ct)
    return packed


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
    assert T <= D, "attention rows are packed into D slots"
    setup_started = time.perf_counter()
    cc = build_context()
    ring = cc.GetRingDimension()
    print(
        f"OpenFHE CKKS context: ring_dim={ring}, mult_depth={DEPTH}, "
        f"128-bit; D={D} T={T} heads={N_HEADS}",
        flush=True,
    )
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, list(range(1, D)))
    setup_keygen_s = time.perf_counter() - setup_started
    print(f"context + keygen: {setup_keygen_s:.1f}s", flush=True)

    # encrypt the (embedded) input tokens
    x_cts = [enc(cc, keys, x[i]) for i in range(T)]

    t0 = time.perf_counter()
    eval_cc = EvalOnlyContext(cc)
    out_cts, level_trace = encrypted_block(eval_cc, x_cts, p)
    assert eval_cc.decrypt_attempts == 0
    packed_out = pack_outputs(eval_cc, out_cts)
    latency = time.perf_counter() - t0

    # Exactly one Decrypt call, after the complete encrypted block and final packing.
    got = dec(cc, keys, packed_out).reshape(T, D)
    levels_before_pack = [int(ct.GetLevel()) for ct in out_cts]
    packed_level = int(packed_out.GetLevel())

    g = gate(got, oracle)
    result = {
        "schema_version": 1,
        "task": "FHE toy DNAGPT block (CKKS/OpenFHE), end-to-end, no mid-decrypt",
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
            "levels_consumed_before_pack": levels_before_pack,
            "packed_output_level": packed_level,
            "level_trace": level_trace,
            "output_packing": (
                "mask folded into final MLP diagonal products; additive pack"
            ),
            "bootstraps": 0,
            "cheb_deg": {
                "invsqrt": DEG_INVSQRT,
                "exp": DEG_EXP,
                "recip": DEG_RECIP,
                "gelu": DEG_GELU,
            },
        },
        "global_rel_inf": g["global_rel_inf"],
        "worst_token_rel_inf": g["worst_token_rel_inf"],
        "tol": g["tol"],
        "passed": g["passed"],
        "context_setup_keygen_s": setup_keygen_s,
        "evaluation_latency_s": latency,
        "latency_label": args.latency_label,
        "intermediate_decrypt_attempts": eval_cc.decrypt_attempts,
        "final_decrypt_calls": 1,
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
        f"1 decryption on the lineage (final read-out only)"
    )
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
