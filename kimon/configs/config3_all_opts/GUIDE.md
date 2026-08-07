# Config 3 — ALL optimizations combined

> New here? `docs/hybrid/t123_walkthrough.md` walks the T123 variant of this job
> end to end (input file → GPU math → client boundaries → final decrypt), explains
> every crypto term, and says what lives on disk / in host RAM / on the GPU.

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
## e2e (all 12 blocks + GSR head, with diagcache) — READY

```bash
mkdir -p kimon/logs/config3_all_opts
sbatch kimon/configs/config3_all_opts/e2e.sbatch     # edit -A; walltime 48h
```
- Driver `real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache`
  (the config-1 e2e driver composing the diagcache block ×12 + head) via runner
  `run_scheme_b_all_blocks_head_t103_cpudiagcache.sh`.
- Fixture: same `multiblock` fixture as config-1 e2e.
- **PASS** = `REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_PASS`.
- Est. **15–45 h**, single GPU. This is the fastest single-GPU full forward pass.
  Compare its `encrypted_evaluation` against config-1 e2e via
  `kimon/analyze/summarize.py` to read off the diagcache speedup at e2e scale.
