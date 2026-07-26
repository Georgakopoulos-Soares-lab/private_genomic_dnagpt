# Brev FHE runbook — Scheme B (hybrid client-assisted CKKS)

Companion to [../pure/brev_runbook.md](../pure/brev_runbook.md) for the frozen Scheme A
gates. This runbook reuses the existing `awesome-gpu-name` workspace. It does not
create, stop, or delete an instance and does not inspect credentials.

## Safety and capacity check

Same as Scheme A — see
[../pure/brev_runbook.md#safety-and-capacity-check](../pure/brev_runbook.md).

## Build

Scheme B forks `fhe/gpu_real_sigmoid`'s FIDESlib source
(`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp`) into its own CMake target,
built against the same pinned FIDESlib commit as every other GPU gate
(`786c7600fb2f16b724e0acf73df367b27b8afed6`) and the same
`dnagpt-fideslib:786c-asymfix2` image:

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh
```

`build_in_fideslib.sh` refuses to build against any other FIDESlib commit and produces
`fhe/gpu_real_scheme_b/build/real_dnagpt_fides_scheme_b`.

## Direct single-gate launch

`run_scheme_b.sh` invokes the built binary directly (used interactively, or as the
payload inside a container by `launch_brev_scheme_b.sh` below). Gate is one of
`ln1`, `attention`, `full`; the output filename must contain `_scheme_b_` (enforced) so
Scheme A and Scheme B evidence can never collide:

```bash
FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b.sh 0 ln1 "$FIXTURE_DIR" \
  /work/results/runs/fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725.json
```

It fails closed on: a GATE other than the three above, an output path missing
`_scheme_b_`, an existing output file (immutable evidence, never overwritten), a
FIDESlib commit mismatch, or a fixture manifest hash mismatch against the pinned
`8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe`.

## Detached launch with GPU preflight

`launch_brev_scheme_b.sh` wraps the same binary in a detached `docker run` on a chosen
physical GPU, with a soft-fail-open preflight (confirms idle only when `nvidia-smi`
gives a clean reading — see the script's own header comment for the shared-host
reliability issue this works around) and refuses to overwrite an existing log/done/
evidence file for the same run tag:

```bash
fhe/gpu_real_scheme_b/launch_brev_scheme_b.sh \
  PHYSICAL_GPU REMOTE_ROOT SOURCE_SUBDIR FIXTURE_SUBDIR IMAGE_TAG GATE RUN_TAG
```

`RUN_TAG` must contain `_scheme_b_`. Poll `${SOURCE_DIR}/${RUN_TAG}.done` for the exit
status and `${SOURCE_DIR}/${RUN_TAG}.run.log` for output.

## Unattended orchestrator

`wait_and_run_scheme_b.sh` runs on the Brev host directly (needs host `uptime`/
`nvidia-smi`), polling host load average and per-GPU idle state every 60s until
capacity is available, then launches the `attention` and `full` gates in sequence via
`launch_brev_scheme_b.sh`. Safe to re-run: it skips any gate whose evidence JSON
already exists and never overwrites an existing log/done/output file.

```bash
fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh
```

`LOAD_THRESHOLD` (default 250), `POLL_SECONDS` (default 60), and `MAX_WAIT_SECONDS`
(default 72h) are overridable via environment variables. Log:
`${SOURCE_DIR}/scheme_b_orchestrator.log` on the host. This is how the `attention` and
`full` block-0 gates in [tasks.md](tasks.md) actually completed — a CPU-bound rotation-
key-generation stall under extreme shared-host contention killed the first interactive
attempt, so the remaining gates were left to the orchestrator to launch once load
dropped.

## Exact commands for the retained numbers

See [tasks.md](tasks.md) for the exact `run_scheme_b.sh` invocations (ln1/attention/full)
that produced each currently-retained Scheme B evidence file.

`context_keygen_load` is service setup; `encrypted_evaluation` is the forward latency,
further split into `server_linear_algebra_seconds` and `client_boundary_seconds_total`.
Do not combine setup and evaluation when extrapolating per-query cost, and do not omit
either from the evidence.
