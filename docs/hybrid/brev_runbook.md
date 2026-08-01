# GPU runbook — client-assisted CKKS

This runbook covers safe execution of the retained FIDESlib/OpenFHE GPU targets. Historical launchers
remain in the repository to reproduce accepted and negative results; their presence does not put them
in the current queue. The active queue is defined only by [roadmap.md](roadmap.md).

The scripts operate on an existing GPU host. They do not create, stop, or delete infrastructure and do
not require inspecting credentials.

## Before a run

1. Confirm the gate is still ordered in [roadmap.md](roadmap.md).
2. Run the local contracts:

   ```bash
   PYTHONPATH=. .venv/bin/python -m unittest discover -s fhe/gpu_real_scheme_b -p "test_*.py"
   ```

3. Confirm the fixture and FIDESlib checkout expected by the versioned launcher are present.
4. Use a new output name. Launchers refuse to overwrite accepted evidence.
5. For performance work, require a dedicated host or record enough system telemetry to demonstrate
   equivalent isolation. An idle GPU poll alone is insufficient because CPU contention dominated prior
   long runs.

## Build

Inside the pinned FIDESlib CUDA environment:

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh
```

The default build includes runnable and historically reproducible targets. The known-unbuildable
process-separated MLP transport prototype is intentionally excluded from the default build because the
pinned backend has no ciphertext serialization API; its source remains only as negative implementation
evidence.

## Current clean-baseline pair

The retained uncached and CPU-vector-cache task-length blocks have matching arithmetic and context
parameters:

```bash
fhe/gpu_real_scheme_b/run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh \
  GPU FIXTURE_DIR OUTPUT_JSON

fhe/gpu_real_scheme_b/run_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh \
  GPU FIXTURE_DIR OUTPUT_JSON
```

Run one warm-up followed by at least three paired repetitions. Record host load, CPU utilization, GPU
utilization, target-process RAM/VRAM, and stage timings. Do not compare runs from different contention
windows as a cache speedup.

The existing `real_dnagpt_fides_scheme_b_profiled` target profiles the older two-token graph. It is
historical evidence, not the profiler for the current 103-token block. Add current-stage NVTX ranges and
capture Nsight Systems before selecting launch, stream, transfer, or kernel work.

## Full-model driver

`real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head` builds and passes local contracts but has no
accepted GPU result. It is intentionally not wired into the old generic unattended orchestrator.

- It may be launched after explicit review to close arithmetic correctness.
- It must not supply the final performance claim until the retained Stage-1/Stage-2 optimizations in
  [roadmap.md](roadmap.md) have been resolved and integrated.
- A quiet-window gate reduces launch risk but cannot guarantee a multi-hour shared host remains clean.
  A dedicated allocation is the required performance environment.

## Historical orchestrators

`wait_and_run_scheme_b*.sh`, `launch_brev_scheme_b*.sh`, and the corresponding versioned run scripts
encode the exact controls used by earlier experiments. Do not use the generic
`wait_and_run_scheme_b.sh` as “run the next job”: its original LN/attention/full queue is complete.

When reproducing a historical number, use the exact command in [tasks.md](tasks.md), including its
matching source and fixture. When creating new evidence, fork the closest retained launcher only after
the new local contract passes.

## Evidence interpretation

- `context_keygen_load` is service setup, not per-query encrypted evaluation.
- `server_linear_algebra_seconds` currently includes host preparation, encoding/upload, GPU work, and
  synchronization unless a finer profile states otherwise.
- `client_boundary_seconds_total` measures local client cryptographic/plaintext work, not WAN latency.
- A cryptographic crossing is not a network RPC until transport is implemented.
- Report target-process memory separately from whole-device memory that includes co-tenants.
