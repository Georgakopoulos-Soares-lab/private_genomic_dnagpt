# Config 3 — ALL optimizations combined

**What it measures:** one full block, T=103, with **every optimization we've
verified accuracy-wise** stacked into a single-GPU run: B=8 Token-SIMD +
depth-13/digits-3/ring-65536 + **CPU-side diagonal-vector cache** (avoids
recomputing the packed diagonal 13× per block). Accuracy-verified to ~`1e-9`.
This is the fastest **single-GPU** configuration.

"Conceptually fastest with no contention": the diagonal cache removes redundant
CPU work; on an uncontended A100 this is the lowest single-GPU latency. The
**additional** win of splitting across GPUs is Config 2 (sharding); the combined
sharding+diagcache block is `block_sharded.sbatch` (optional, see below).

## Prerequisites
Same as Config 1.

## Run
From the repo root:
```bash
mkdir -p kimon/logs/config3_all_opts
sbatch kimon/configs/config3_all_opts/block.sbatch      # edit -A first
```

## Output
`kimon/logs/config3_all_opts/<tag>/<tag>.json` + `run.log`.
**PASS** = log contains `REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_GATE_PASS`.
The JSON reports `diagonal_cache_hits`/`diagonal_cache_misses` (expect ~99.99% hit)
alongside timings and `global_rel_inf`.

## Notes
- Binary: `real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache`.
- Fixture: same `block` fixture as Config 1.
- Timing depends heavily on host CPU contention (the diagonal cache is CPU-side).
  Run on an uncontended node for a meaningful number.
- `block_sharded.sbatch` (sharding + diagcache) uses the combined reader
  `..._simd_shard_reader_cpudiagcache_t103_depth13` — provided if that binary
  built; otherwise use Config 2 (sharding) and Config 3 (diagcache) separately.
- e2e (diagcache 12-block): **being built** — see `../../README.md` §e2e status.
