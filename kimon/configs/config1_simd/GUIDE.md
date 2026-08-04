# Config 1 — SIMD batching (baseline optimization)

**What it measures:** one full DNAGPT transformer block under FHE with **B=8
Token-SIMD packing** at depth-13 / digits-3 / ring-65536, T=103 tokens, on **1
GPU**. This is the SIMD-batching baseline the other configs build on.

## Prerequisites (once)
See `../../README.md` §Setup. In short, from the repo root:
1. `kimon/env/build_sif.sh archive fideslib.tar` → builds `fideslib.sif`
2. `kimon/env/build_binaries.sh` → compiles the run binaries
3. `kimon/env/prepare_fixtures.sh` → ensures the `block` fixture exists

## Run
From the **repo root**:
```bash
mkdir -p kimon/logs/config1_simd
sbatch kimon/configs/config1_simd/block.sbatch      # edit -A CHANGE_ME_ALLOCATION first
```
Native mode (FIDESlib preinstalled, no apptainer): prepend `RUN_MODE=native`.

## Output
`kimon/logs/config1_simd/<tag>/`:
- `<tag>.json` — the immutable result (timings, `global_rel_inf`, `passed`).
- `run.log` — full stdout.

**PASS** = the log contains `REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_GATE_PASS`
and `global_rel_inf`/`worst_token_rel_inf` ≤ `4e-2` (typically ~`1e-9`).

## Notes
- Binary: `real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536`.
- Fixture: `checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0`.
- Reference latency (uncontended A100, historical): ~4600 s encrypted eval for
  one block. Real per-job GPU memory footprint ~10 GB (fits the 40 GB A100).
## e2e (all 12 blocks + GSR head) — READY

```bash
mkdir -p kimon/logs/config1_simd
sbatch kimon/configs/config1_simd/e2e.sbatch        # edit -A; walltime 48h
```
- Binary `real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head` via runner
  `run_scheme_b_all_blocks_head_t103_simd.sh` (authored for TACC).
- Fixture: `checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0`
  (the `multiblock` fixture — `prepare_fixtures.sh` generates it).
- **PASS** = `REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_SIMD_PASS`.
- Est. **15.5–45.5 h**, single GPU. First real run of this driver.
