# -*- coding: utf-8 -*-
"""Prove that OpenFHE CKKS bootstrapping refreshes a depleted ciphertext.

The test encrypts near the bottom of the modulus chain, bootstraps, consumes more
levels than were available before refresh, and decrypts once at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

NUM_SLOTS = 8
# OpenFHE clamps [4,4] to [3,3] at this eight-slot toy size. Declare the
# effective budget directly so the recorded configuration matches execution.
LEVEL_BUDGET = [3, 3]
LEVELS_AFTER_BOOTSTRAP = 10
TOL = 4e-2


def _decrypt(cc, keys, ct):
    try:
        plaintext = cc.Decrypt(ct, keys.secretKey)
    except Exception:
        plaintext = cc.Decrypt(keys.secretKey, ct)
    plaintext.SetLength(NUM_SLOTS)
    return np.array(plaintext.GetRealPackedValue())


def run():
    from openfhe import (
        CCParamsCKKSRNS,
        FHECKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        ScalingTechnique,
        SecretKeyDist,
        SecurityLevel,
    )

    secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
    params = CCParamsCKKSRNS()
    params.SetSecretKeyDist(secret_key_dist)
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetScalingModSize(59)
    params.SetFirstModSize(60)
    bootstrap_depth = FHECKKSRNS.GetBootstrapDepth(LEVEL_BUDGET, secret_key_dist)
    depth = LEVELS_AFTER_BOOTSTRAP + bootstrap_depth
    params.SetMultiplicativeDepth(depth)

    cc = GenCryptoContext(params)
    for feature in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.KEYSWITCH,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
        PKESchemeFeature.FHE,
    ):
        cc.Enable(feature)

    setup_started = time.perf_counter()
    cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], NUM_SLOTS)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, NUM_SLOTS)
    setup_keygen_s = time.perf_counter() - setup_started

    rng = np.random.default_rng(0)
    reference = rng.standard_normal(NUM_SLOTS) * 0.5
    near_exhausted_pt = cc.MakeCKKSPackedPlaintext(
        reference.tolist(), 1, depth - 1, None, NUM_SLOTS
    )
    ct = cc.Encrypt(keys.publicKey, near_exhausted_pt)
    level_before = int(ct.GetLevel())

    started = time.perf_counter()
    ct = cc.EvalBootstrap(ct)
    bootstrap_s = time.perf_counter() - started
    level_after = int(ct.GetLevel())

    ones = cc.MakeCKKSPackedPlaintext([1.0] * NUM_SLOTS, 1, 0, None, NUM_SLOTS)
    extra_levels = 0
    while ct.GetLevel() < depth - 2:
        ct = cc.EvalMult(ct, ones)
        extra_levels += 1

    got = _decrypt(cc, keys, ct)
    max_abs_err = float(np.max(np.abs(got - reference)))
    denom = float(np.max(np.abs(reference)))
    rel_inf = max_abs_err / denom
    passed = bool(
        np.all(np.isfinite(got))
        and level_after < level_before
        and extra_levels > 1
        and rel_inf <= TOL
    )
    return {
        "schema_version": 1,
        "task": "OpenFHE CKKS bootstrap refresh",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "OpenFHE CKKS",
        "security": "HEStd_128_classic",
        "environment": "Docker linux/amd64 on Apple Silicon [emu]",
        "config": {
            "slots": NUM_SLOTS,
            "ring_dim": int(cc.GetRingDimension()),
            "multiplicative_depth": int(depth),
            "bootstrap_depth": int(bootstrap_depth),
            "level_budget": LEVEL_BUDGET,
        },
        "setup_keygen_s": setup_keygen_s,
        "bootstrap_latency_emu_s": bootstrap_s,
        "level_before_bootstrap": level_before,
        "level_after_bootstrap": level_after,
        "levels_restored": level_before - level_after,
        "post_bootstrap_levels_consumed": extra_levels,
        "final_level": int(ct.GetLevel()),
        "max_abs_err": max_abs_err,
        "rel_inf": rel_inf,
        "tol": TOL,
        "final_decrypt_calls": 1,
        "passed": passed,
    }


def _write_immutable(result, tag):
    path = os.path.join("results", "runs", f"{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable run: {path}")
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="write immutable results/runs/<tag>.json")
    args = parser.parse_args()
    try:
        import openfhe  # noqa: F401
    except Exception:
        print("[skip] openfhe absent — run in the dnagpt-openfhe Docker image.")
        return 0

    result = run()
    print(json.dumps(result, indent=2))
    if args.tag:
        print(f"[evidence] {_write_immutable(result, args.tag)}")
    print(
        "[pass] bootstrap restored usable depth"
        if result["passed"]
        else "[FAIL] bootstrap refresh gate failed"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
