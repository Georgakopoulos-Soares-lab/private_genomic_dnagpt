# -*- coding: utf-8 -*-
"""Self-check DNAGPT's CKKS operator set under OpenFHE at 128-bit security.

Each operator starts from a fresh ciphertext, decrypts only its final result for
validation, and reports evaluation-only latency with an explicit environment label.
The combined run can write one immutable evidence JSON:

  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
    dnagpt-openfhe python3 -u fhe/ops/op_matrix_ckks.py \
    --tag fhe_operator_matrix_d8_20260724
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

D = 8
DEPTH = 20
SCALE_BITS = 50
TOL = 4e-2
DEG = {"invsqrt": 27, "exp": 13, "recip": 27}
GELU_PROFILES = {
    "broad": {"domain": (-4.0, 4.0), "degree": 13},
    "toy": {"domain": (-2.0, 2.0), "degree": 7},
}
OPS = ("matmul", "layernorm", "softmax", "gelu")


def _dec(cc, keys, ct, n=D):
    try:
        plaintext = cc.Decrypt(ct, keys.secretKey)
    except Exception:
        plaintext = cc.Decrypt(keys.secretKey, ct)
    plaintext.SetLength(n)
    return np.array(plaintext.GetRealPackedValue())


def _bsgs_plan(n):
    n1 = math.isqrt(n)
    while n % n1:
        n1 += 1
    return n1, n // n1


def _bsgs_matmul(cc, ct, matrix):
    n = matrix.shape[0]
    diagonals = [
        np.array([matrix[i, (i + diagonal) % n] for i in range(n)])
        for diagonal in range(n)
    ]
    n1, n2 = _bsgs_plan(n)
    baby = {0: ct}
    for j in range(1, n1):
        baby[j] = cc.EvalRotate(ct, j)
    result = None
    for k in range(n2):
        inner = None
        for j in range(n1):
            diagonal = np.roll(diagonals[n1 * k + j], n1 * k)
            term = cc.EvalMult(baby[j], cc.MakeCKKSPackedPlaintext(diagonal.tolist()))
            inner = term if inner is None else cc.EvalAdd(inner, term)
        if k:
            inner = cc.EvalRotate(inner, n1 * k)
        result = inner if result is None else cc.EvalAdd(result, inner)
    return result


def _build_context():
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        SecurityLevel,
    )

    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetMultiplicativeDepth(DEPTH)
    params.SetScalingModSize(SCALE_BITS)
    params.SetBatchSize(D)
    cc = GenCryptoContext(params)
    for feature in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.KEYSWITCH,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
    ):
        cc.Enable(feature)
    started = time.perf_counter()
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, list(range(1, D)))
    return cc, keys, time.perf_counter() - started


def run(
    selected=None,
    environment="Docker linux/amd64 on Apple Silicon [emu]",
    latency_label="[emu]",
    gelu_profile="broad",
):
    """Run selected operator names and return a serializable evidence object."""
    selected = OPS if selected is None else tuple(selected)
    unknown = set(selected) - set(OPS)
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    if gelu_profile not in GELU_PROFILES:
        raise ValueError(f"unknown GELU profile: {gelu_profile}")

    cc, keys, keygen_s = _build_context()
    make_pt = cc.MakeCKKSPackedPlaintext

    def encrypt(values):
        return cc.Encrypt(keys.publicKey, make_pt(list(values)))

    rng = np.random.default_rng(0)
    rows = []

    def record(name, ct, reference, started, domain=None, algorithm=None):
        got = _dec(cc, keys, ct)[: len(reference)]
        max_abs_err = float(np.max(np.abs(got - reference)))
        denom = float(np.max(np.abs(reference)))
        rel_inf = float(max_abs_err / denom) if denom else float("inf")
        row = {
            "op": name,
            "levels_consumed": int(ct.GetLevel()),
            "max_abs_err": max_abs_err,
            "rel_inf": rel_inf,
            "tol": TOL,
            "passed": bool(np.all(np.isfinite(got)) and rel_inf <= TOL),
            "latency_s": time.perf_counter() - started,
            "latency_label": latency_label,
        }
        if domain is not None:
            row["public_polynomial_domain"] = domain
        if algorithm is not None:
            row["algorithm"] = algorithm
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Generate all inputs in a fixed order so --only runs reproduce combined-run values.
    matmul_w = rng.standard_normal((D, D)) * (0.5 / math.sqrt(D))
    matmul_x = rng.standard_normal(D) * 0.5
    layernorm_x = rng.standard_normal(D) * 0.5
    softmax_x = rng.standard_normal(D) * 0.5
    gelu_x = rng.standard_normal(D)

    if "matmul" in selected:
        ct = encrypt(matmul_x)
        started = time.perf_counter()
        out = _bsgs_matmul(cc, ct, matmul_w)
        n1, n2 = _bsgs_plan(D)
        record(
            "linear/matmul",
            out,
            matmul_w @ matmul_x,
            started,
            algorithm={
                "name": "Halevi-Shoup diagonal BSGS",
                "baby_steps": n1,
                "giant_steps": n2,
                "ciphertext_rotations": (n1 - 1) + (n2 - 1),
            },
        )

    if "layernorm" in selected:
        ct = encrypt(layernorm_x)
        started = time.perf_counter()
        mu = cc.EvalMult(cc.EvalSum(ct, D), make_pt([1.0 / D] * D))
        centered = cc.EvalSub(ct, mu)
        variance = cc.EvalAdd(
            cc.EvalMult(
                cc.EvalSum(cc.EvalMult(centered, centered), D),
                make_pt([1.0 / D] * D),
            ),
            make_pt([1e-5] * D),
        )
        inv = cc.EvalChebyshevFunction(
            lambda value: 1 / math.sqrt(value),
            variance,
            0.03,
            0.9,
            DEG["invsqrt"],
        )
        out = cc.EvalMult(centered, inv)
        reference = (layernorm_x - layernorm_x.mean()) / math.sqrt(
            layernorm_x.var() + 1e-5
        )
        record(
            "layernorm",
            out,
            reference,
            started,
            domain={"invsqrt": [0.03, 0.9], "degree": DEG["invsqrt"]},
        )

    if "softmax" in selected:
        ct = encrypt(softmax_x)
        started = time.perf_counter()
        exp_ct = cc.EvalChebyshevFunction(math.exp, ct, -2.0, 2.0, DEG["exp"])
        exp_sum = cc.EvalSum(exp_ct, D)
        inv = cc.EvalDivide(exp_sum, 0.5, 12.0, DEG["recip"])
        out = cc.EvalMult(exp_ct, inv)
        reference = np.exp(softmax_x) / np.exp(softmax_x).sum()
        record(
            "causal-softmax primitive",
            out,
            reference,
            started,
            domain={
                "exp": [-2.0, 2.0],
                "exp_degree": DEG["exp"],
                "reciprocal": [0.5, 12.0],
                "reciprocal_degree": DEG["recip"],
            },
        )

    if "gelu" in selected:
        gelu_domain = GELU_PROFILES[gelu_profile]["domain"]
        gelu_degree = GELU_PROFILES[gelu_profile]["degree"]
        ct = encrypt(gelu_x)
        started = time.perf_counter()

        def gelu_fn(value):
            return (
                0.5
                * value
                * (
                    1
                    + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3))
                )
            )

        out = cc.EvalChebyshevFunction(
            gelu_fn, ct, gelu_domain[0], gelu_domain[1], gelu_degree
        )
        reference = (
            0.5
            * gelu_x
            * (1 + np.tanh(math.sqrt(2 / math.pi) * (gelu_x + 0.044715 * gelu_x**3)))
        )
        record(
            "gelu(tanh)",
            out,
            reference,
            started,
            domain={"gelu": list(gelu_domain), "degree": gelu_degree},
        )

    return {
        "schema_version": 2,
        "task": "DNAGPT CKKS operator feasibility matrix",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "OpenFHE CKKS",
        "security": "HEStd_128_classic",
        "environment": environment,
        "config": {
            "dimension": D,
            "ring_dim": int(cc.GetRingDimension()),
            "multiplicative_depth": DEPTH,
            "scaling_mod_bits": SCALE_BITS,
            "context_keygen_s": keygen_s,
        },
        "operators": rows,
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def _write_immutable(result, tag):
    out_path = os.path.join("results", "runs", f"{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(f"refusing to overwrite immutable run: {out_path}")
    with open(out_path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return out_path


def main(default_only=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=OPS, default=default_only)
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
    parser.add_argument(
        "--gelu-profile",
        choices=tuple(GELU_PROFILES),
        default="broad",
        help="broad reproduces the operator matrix; toy matches the block domain",
    )
    args = parser.parse_args()
    selected = (args.only,) if args.only else None

    try:
        import openfhe  # noqa: F401
    except Exception:
        print("[skip] openfhe absent — run in the dnagpt-openfhe Docker image.")
        return 0

    result = run(
        selected,
        environment=args.environment,
        latency_label=args.latency_label,
        gelu_profile=args.gelu_profile,
    )
    print(json.dumps(result, indent=2))
    if args.tag:
        path = _write_immutable(result, args.tag)
        print(f"[evidence] {path}")
    print(
        "[pass] requested operators are within tolerance"
        if result["passed"]
        else "[FAIL] operator tolerance exceeded"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
