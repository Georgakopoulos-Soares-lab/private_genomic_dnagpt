# -*- coding: utf-8 -*-
"""Correctness-first CKKS optimization probes for the DNAGPT GPU path.

This script tests scheduling changes that can be validated before writing CUDA:

1. diagonal matmul: naive rotations vs BSGS vs BSGS with hoisted baby rotations;
2. attention aggregation: materialized softmax vs numerator-first scheduling; and
3. GELU polynomial degree selection on the toy block's public activation domain.

Every row is an independent encrypted circuit. Decryption is used only after that
row's encrypted output has been produced, solely for comparison with a NumPy oracle.
Mac Docker timings are directional emulation measurements, not GPU predictions.

Run:
  docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
    dnagpt-openfhe python3 -u fhe/optimizations/local_ckks_probe.py \
    --tag fhe_local_optimizations_20260724
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MATMUL_D = 16
ATTN_D = 8
ATTN_T = 4
SCALE_BITS = 50
TOL = 4e-2


def _decrypt(cc, keys, ct, length):
    try:
        plaintext = cc.Decrypt(ct, keys.secretKey)
    except Exception:
        plaintext = cc.Decrypt(keys.secretKey, ct)
    plaintext.SetLength(length)
    return np.asarray(plaintext.GetRealPackedValue()[:length], dtype=np.float64)


def _rel_inf(got, reference):
    got = np.asarray(got)
    reference = np.asarray(reference)
    denominator = float(np.max(np.abs(reference)))
    error = float(np.max(np.abs(got - reference)))
    return error / denominator if denominator else error


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _context(batch_size, depth, rotation_indexes, eval_sum=False):
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        SecurityLevel,
    )

    params = CCParamsCKKSRNS()
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(SCALE_BITS)
    params.SetBatchSize(batch_size)
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
    if eval_sum:
        cc.EvalSumKeyGen(keys.secretKey)
    if rotation_indexes:
        cc.EvalRotateKeyGen(keys.secretKey, sorted(set(rotation_indexes)))
    return cc, keys, time.perf_counter() - started


def _diagonals(matrix):
    n = matrix.shape[0]
    return [
        np.asarray([matrix[row, (row + diagonal) % n] for row in range(n)])
        for diagonal in range(n)
    ]


def _bsgs_plan(n):
    n1 = math.isqrt(n)
    while n % n1:
        n1 += 1
    return n1, n // n1


def _encode_matmul_plaintexts(cc, matrix):
    n = matrix.shape[0]
    diagonals = _diagonals(matrix)
    naive = [cc.MakeCKKSPackedPlaintext(diagonal.tolist()) for diagonal in diagonals]
    n1, n2 = _bsgs_plan(n)
    bsgs = [
        [
            cc.MakeCKKSPackedPlaintext(np.roll(diagonals[n1 * k + j], n1 * k).tolist())
            for j in range(n1)
        ]
        for k in range(n2)
    ]
    return naive, bsgs, n1, n2


def _naive_matmul(cc, ct, plaintext_diagonals):
    result = cc.EvalMult(ct, plaintext_diagonals[0])
    for index in range(1, len(plaintext_diagonals)):
        term = cc.EvalMult(cc.EvalRotate(ct, index), plaintext_diagonals[index])
        result = cc.EvalAdd(result, term)
    return result


def _bsgs_matmul(cc, ct, plaintext_diagonals, n1, n2, hoisted):
    baby = {0: ct}
    if hoisted:
        digits = cc.EvalFastRotationPrecompute(ct)
        for index in range(1, n1):
            baby[index] = cc.EvalFastRotation(ct, index, digits)
    else:
        for index in range(1, n1):
            baby[index] = cc.EvalRotate(ct, index)

    result = None
    for k in range(n2):
        inner = None
        for j in range(n1):
            term = cc.EvalMult(baby[j], plaintext_diagonals[k][j])
            inner = term if inner is None else cc.EvalAdd(inner, term)
        if k:
            inner = cc.EvalRotate(inner, n1 * k)
        result = inner if result is None else cc.EvalAdd(result, inner)
    return result


def _run_matmul_probe(environment, latency_label):
    n1, n2 = _bsgs_plan(MATMUL_D)
    rotations = list(range(1, MATMUL_D))
    cc, keys, keygen_s = _context(MATMUL_D, 5, rotations)
    rng = np.random.default_rng(20260724)
    matrix = rng.standard_normal((MATMUL_D, MATMUL_D)) / math.sqrt(MATMUL_D)
    vector = rng.standard_normal(MATMUL_D) * 0.5
    reference = matrix @ vector
    make_pt = cc.MakeCKKSPackedPlaintext
    ct = cc.Encrypt(keys.publicKey, make_pt(vector.tolist()))
    naive_pt, bsgs_pt, n1, n2 = _encode_matmul_plaintexts(cc, matrix)

    variants = (
        (
            "naive_diagonal",
            lambda: _naive_matmul(cc, ct, naive_pt),
            MATMUL_D - 1,
            MATMUL_D - 1,
        ),
        (
            "bsgs",
            lambda: _bsgs_matmul(cc, ct, bsgs_pt, n1, n2, False),
            (n1 - 1) + (n2 - 1),
            (n1 - 1) + (n2 - 1),
        ),
        (
            "bsgs_hoisted_baby_rotations",
            lambda: _bsgs_matmul(cc, ct, bsgs_pt, n1, n2, True),
            (n1 - 1) + (n2 - 1),
            1 + (n2 - 1),
        ),
    )

    rows = []
    for name, evaluate, rotation_count, decomposition_count in variants:
        started = time.perf_counter()
        out = evaluate()
        latency_s = time.perf_counter() - started
        got = _decrypt(cc, keys, out, MATMUL_D)
        rel_inf = _rel_inf(got, reference)
        rows.append(
            {
                "variant": name,
                "dimension": MATMUL_D,
                "bsgs_plan": [n1, n2],
                "ciphertext_rotations": rotation_count,
                "key_switch_decompositions": decomposition_count,
                "levels_consumed": int(out.GetLevel()),
                "rel_inf": rel_inf,
                "tol": TOL,
                "passed": bool(np.all(np.isfinite(got)) and rel_inf <= TOL),
                "evaluation_seconds": latency_s,
                "latency_label": latency_label,
            }
        )
        print(json.dumps({"probe": "matmul", **rows[-1]}), flush=True)

    return {
        "environment": environment,
        "keygen_seconds": keygen_s,
        "plaintext_encoding_excluded_from_evaluation_timing": True,
        "rows": rows,
        "correctness_passed": all(row["passed"] for row in rows),
        "operation_reduction": {
            "naive_to_bsgs_rotations": {
                "from": MATMUL_D - 1,
                "to": (n1 - 1) + (n2 - 1),
            },
            "bsgs_to_hoisted_decompositions": {
                "from": (n1 - 1) + (n2 - 1),
                "to": 1 + (n2 - 1),
            },
        },
    }


def _gelu(value):
    return (
        0.5
        * value
        * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
    )


def _run_gelu_probe(environment, latency_label):
    degrees = (3, 5, 7, 9, 13)
    cc, keys, keygen_s = _context(ATTN_D, 16, [])
    values = np.linspace(-2.0, 2.0, ATTN_D)
    reference = np.asarray([_gelu(value) for value in values])
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(values.tolist()))
    rows = []
    for degree in degrees:
        started = time.perf_counter()
        out = cc.EvalChebyshevFunction(_gelu, ct, -2.0, 2.0, degree)
        latency_s = time.perf_counter() - started
        got = _decrypt(cc, keys, out, ATTN_D)
        rel_inf = _rel_inf(got, reference)
        rows.append(
            {
                "degree": degree,
                "public_domain": [-2.0, 2.0],
                "levels_consumed": int(out.GetLevel()),
                "rel_inf": rel_inf,
                "tol": TOL,
                "passed": bool(np.all(np.isfinite(got)) and rel_inf <= TOL),
                "evaluation_seconds": latency_s,
                "latency_label": latency_label,
            }
        )
        print(json.dumps({"probe": "gelu_degree", **rows[-1]}), flush=True)

    passing = [row["degree"] for row in rows if row["passed"]]
    return {
        "environment": environment,
        "keygen_seconds": keygen_s,
        "rows": rows,
        "lowest_passing_degree_on_sample_grid": min(passing) if passing else None,
        "correctness_passed": bool(passing),
        "scope": "sample-grid screen; the complete block oracle remains load-bearing",
    }


def _run_attention_probe(environment, latency_label):
    cc, keys, keygen_s = _context(ATTN_D, 24, list(range(1, ATTN_D)), True)
    make_pt = cc.MakeCKKSPackedPlaintext
    rng = np.random.default_rng(7)
    exponentials = np.exp(rng.uniform(-0.8, 0.8, ATTN_T))
    values = rng.standard_normal((ATTN_T, ATTN_D)) * 0.4
    reference = exponentials @ values / exponentials.sum()
    padded_exp = np.pad(exponentials, (0, ATTN_D - ATTN_T))
    exp_ct = cc.Encrypt(keys.publicKey, make_pt(padded_exp.tolist()))
    value_cts = [cc.Encrypt(keys.publicKey, make_pt(row.tolist())) for row in values]
    masks = [
        make_pt(([0.0] * index) + [1.0] + ([0.0] * (ATTN_D - index - 1)))
        for index in range(ATTN_T)
    ]

    def inverse_denominator():
        denominator = cc.EvalSum(exp_ct, ATTN_D)
        return cc.EvalDivide(denominator, 0.5, 12.0, 27)

    def select_and_broadcast(ciphertext, mask):
        return cc.EvalSum(cc.EvalMult(ciphertext, mask), ATTN_D)

    started = time.perf_counter()
    inverse = inverse_denominator()
    softmax = cc.EvalMult(exp_ct, inverse)
    standard = None
    for index in range(ATTN_T):
        weight = select_and_broadcast(softmax, masks[index])
        term = cc.EvalMult(weight, value_cts[index])
        standard = term if standard is None else cc.EvalAdd(standard, term)
    standard_s = time.perf_counter() - started

    started = time.perf_counter()
    inverse = inverse_denominator()
    numerator = None
    for index in range(ATTN_T):
        weight = select_and_broadcast(exp_ct, masks[index])
        term = cc.EvalMult(weight, value_cts[index])
        numerator = term if numerator is None else cc.EvalAdd(numerator, term)
    numerator_first = cc.EvalMult(numerator, inverse)
    numerator_s = time.perf_counter() - started

    standard_got = _decrypt(cc, keys, standard, ATTN_D)
    numerator_got = _decrypt(cc, keys, numerator_first, ATTN_D)
    standard_rel = _rel_inf(standard_got, reference)
    numerator_rel = _rel_inf(numerator_got, reference)
    parity_rel = _rel_inf(numerator_got, standard_got)
    rows = [
        {
            "variant": "materialized_softmax",
            "levels_consumed": int(standard.GetLevel()),
            "rel_inf": standard_rel,
            "evaluation_seconds": standard_s,
            "latency_label": latency_label,
        },
        {
            "variant": "numerator_first",
            "levels_consumed": int(numerator_first.GetLevel()),
            "rel_inf": numerator_rel,
            "evaluation_seconds": numerator_s,
            "latency_label": latency_label,
        },
    ]
    for row in rows:
        row["tol"] = TOL
        row["passed"] = bool(np.isfinite(row["rel_inf"]) and row["rel_inf"] <= TOL)
        print(json.dumps({"probe": "attention_schedule", **row}), flush=True)

    return {
        "environment": environment,
        "keygen_seconds": keygen_s,
        "input_scope": (
            "encrypted exponentials and encrypted value vectors; the exp polynomial "
            "is intentionally held constant outside this schedule comparison"
        ),
        "rows": rows,
        "numerator_first_vs_materialized_rel_inf": parity_rel,
        "levels_saved": int(standard.GetLevel() - numerator_first.GetLevel()),
        "correctness_passed": bool(
            all(row["passed"] for row in rows)
            and np.isfinite(parity_rel)
            and parity_rel <= TOL
        ),
    }


def run(environment, latency_label):
    import openfhe

    started = time.perf_counter()
    matmul = _run_matmul_probe(environment, latency_label)
    attention = _run_attention_probe(environment, latency_label)
    gelu = _run_gelu_probe(environment, latency_label)
    passed = bool(
        matmul["correctness_passed"]
        and attention["correctness_passed"]
        and gelu["correctness_passed"]
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "[V] PASS" if passed else "[V] FAIL",
        "purpose": "local correctness/depth screen for GPU-targeted CKKS optimizations",
        "backend": {
            "scheme": "CKKS",
            "library": "OpenFHE Python",
            "version": getattr(openfhe, "__version__", "unknown"),
            "security": "HEStd_128_classic",
            "scale_bits": SCALE_BITS,
        },
        "environment": {
            "description": environment,
            "latency_label": latency_label,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "gate": {
            "all_finite": True,
            "rel_inf_max": TOL,
            "note": (
                "The local probes are screening gates. Any selected change must also "
                "pass the complete encrypted block oracle before a performance claim."
            ),
        },
        "probes": {
            "matmul": matmul,
            "attention": attention,
            "gelu": gelu,
        },
        "sources": [
            {
                "claim": "hoisted automorphisms reuse digit decomposition",
                "url": (
                    "https://github.com/openfheorg/openfhe-development/blob/main/"
                    "src/pke/include/cryptocontext.h"
                ),
            },
            {
                "claim": "GPU CKKS implementation supports hoisting, fusion, and bootstrapping",
                "url": "https://github.com/CAPS-UMU/FIDESlib",
            },
            {
                "claim": "parallel BSGS trades packing/parallelism for fewer rotations",
                "url": "https://eprint.iacr.org/2024/883",
            },
        ],
        "source_sha256": _source_sha256(),
        "total_seconds": time.perf_counter() - started,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument(
        "--environment",
        default="Docker linux/amd64 on Apple Silicon [emu-directional]",
    )
    parser.add_argument("--latency-label", default="[emu-directional]")
    args = parser.parse_args()

    result = run(args.environment, args.latency_label)
    print(json.dumps({"summary": result["status"], "passed": result["passed"]}))
    if args.tag:
        os.makedirs("results/runs", exist_ok=True)
        output = Path("results/runs") / f"{args.tag}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable result: {output}")
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {output}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
