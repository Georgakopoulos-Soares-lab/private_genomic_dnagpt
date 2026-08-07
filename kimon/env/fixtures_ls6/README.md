# Platform-tagged fixture contracts (TACC Lonestar6)

These `.sha256` files are the **platform-tagged** counterparts of the frozen
contracts in `fhe/gpu_real_scheme_b/` (`fixture_t103.sha256`,
`fixture_t103_qkvshard.sha256`, `fixture_all_blocks_head_t103.sha256`).

**Why they exist.** The frozen contracts were generated on the fixture author's
machine (Mac / Accelerate BLAS). On the ls6 `gpu-a100-small` node the fixtures
regenerate **numerically correct and deterministic**, but a few float64
`@`-matmul *oracle/reference* arrays (e.g. `oracle__attention_projection`,
`oracle__block_output`) differ in the last bits from the Mac-origin hashes —
an unavoidable cross-platform float64 matmul difference (confirmed: even the
exact bundled `scipy-openblas64`, single-threaded, does not reproduce the
Mac bytes). The arrays the encrypted C++ actually consumes — every
`weights__*` and `input__embeddings` — are byte-identical to the frozen
contract. See memory `tacc-ls6-env-quirks`.

**What they are.** Same filename lists as the frozen contracts, with this
node's hashes, regenerated 2026-08-04 with:
- numpy 2.5.1 built from source (LAHF-patched, see `../build_numpy_patched.sh`)
  linked against `scipy-openblas64` 0.3.34,
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`,
- torch 2.13.0 CPU float32.

They are **new evidence**, not a replacement: nothing frozen is overwritten.
`kimon/env/common.sh` points the (opt-in `KIMON_FIXTURE_CONTRACT`) run-script
override at these when `RUN_MODE=native` (default on this node); set
`KIMON_USE_PLATFORM_CONTRACT=0` to force the frozen contracts instead.

Regenerate with: `kimon/env/prepare_fixtures.sh` then
`kimon/env/refresh_platform_contracts.sh` (records `expected.env`).
