# kimon/ — run the encrypted DNAGPT benchmarks on TACC (3× A100)

Hi Kimonas 👋 — this folder is everything you need to run our encrypted
("Scheme B") DNAGPT benchmarks on a TACC GPU node. **Follow the steps top to
bottom.** Every command is copy-paste and is run **from the repo root**
(`.../dna-gpt/`) unless noted.

You do **not** need docker, and you do **not** need to understand the crypto.
There are 3 experiments ("configs"); each has a 1-block run and (mostly) a full
end-to-end run.

---

## 0. What you need before starting

Two things are **not in git** (too big / derived). Get them from Christos:

| Item | What it is | Where to put it |
|---|---|---|
| `fideslib.tar` | the container image (`docker save` of our build) | anywhere; you pass its path in Step 2 |
| 2 fixture folders | the encrypted model inputs (`gsr_pos0_block0_t103_…` and `gsr_pos0_multiblock_t103_…`) | `checkpoints/fhe_exports/` in the repo |

On TACC you also need: an **allocation** (the `-A` account you charge to) and a
**GPU node with 3× A100** (e.g. Lonestar6 partition `gpu-a100`). Load apptainer
once per shell:

```bash
module load tacc-apptainer     # if that name fails, try: module load apptainer
```

---

## 1. Quickstart (the whole thing, in order)

```bash
cd <repo>/dna-gpt                       # always start here
module load tacc-apptainer

# --- one-time setup ---
kimon/env/build_sif.sh archive /path/to/fideslib.tar   # ~5-15 min -> kimon/env/fideslib.sif
kimon/env/prepare_fixtures.sh                          # checks/creates the 2 fixtures
idev -p gpu-a100 -N 1 -t 01:00:00                      # grab an interactive GPU node...
  kimon/env/build_binaries.sh                          # ...then compile our binaries (~15-30 min); exit the node after
exit

# --- edit ONE line in each run you want, then submit ---
# open the .sbatch and set:  #SBATCH -A  <your-allocation>
mkdir -p kimon/logs/config1_simd && sbatch kimon/configs/config1_simd/block.sbatch
```

That's it. Details and the other configs are below. If a step fails, see
§Troubleshooting.

---

## 2. One-time setup (explained)

**Step A — build the container image (`fideslib.sif`).**
```bash
kimon/env/build_sif.sh archive /path/to/fideslib.tar
```
This turns Christos's `fideslib.tar` into `kimon/env/fideslib.sif`. If you have
no tar but the machine has docker, use `daemon`. If you have neither, use `def`
(builds from scratch, ~30-60 min, needs internet). Output: `kimon/env/fideslib.sif`.

**Step B — provision the fixtures.**
```bash
kimon/env/prepare_fixtures.sh
```
If the two fixture folders are already under `checkpoints/fhe_exports/`, it says
`[OK]` and does nothing. If missing and this machine has our weights + `.venv`,
it regenerates them. Otherwise it prints the exact `scp` command to copy them.

**Step C — compile the binaries (needs a GPU).**
```bash
idev -p gpu-a100 -N 1 -t 01:00:00      # interactive node; wait for the prompt
kimon/env/build_binaries.sh            # ~15-30 min, builds only the 6 kimon binaries
exit
```
Binaries land in `fhe/gpu_real_scheme_b/build/`. They are tied to `fideslib.sif`
— rebuild them if you rebuild the .sif.

> **Native mode** (only if FIDESlib is already installed system-wide, no
> apptainer): prepend `RUN_MODE=native` to Steps C and every `sbatch` below.

---

## 3. Run the experiments

**Before submitting any run:** open the `.sbatch` file and set the account line
`#SBATCH -A CHANGE_ME_ALLOCATION` to your TACC allocation. If your GPU partition
isn't `gpu-a100`, also fix `#SBATCH -p`.

| Experiment | Command (from repo root) | GPUs | ~time |
|---|---|---|---|
| **1. SIMD — 1 block** | `mkdir -p kimon/logs/config1_simd && sbatch kimon/configs/config1_simd/block.sbatch` | 1 | ~1-2 h |
| **1. SIMD — end-to-end** | `sbatch kimon/configs/config1_simd/e2e.sbatch` | 1 | 15-45 h |
| **2. Sharding — Q/K/V projection** | `mkdir -p kimon/logs/config2_sharding && sbatch kimon/configs/config2_sharding/block.sbatch` | 2 | ~20 min |
| **3. All-opts — 1 block** | `mkdir -p kimon/logs/config3_all_opts && sbatch kimon/configs/config3_all_opts/block.sbatch` | 1 | ~1 h |
| **3. All-opts — end-to-end** | `sbatch kimon/configs/config3_all_opts/e2e.sbatch` | 1 | 15-45 h |

Each config folder has a `GUIDE.md` explaining exactly what it measures. Read it
if you want the details — you don't need to, to run.

Check a job: `squeue -u $USER`. It will `[PASS]`/`[FAIL]` at the end of its log.

### No SLURM? Run on any GPU box (general / production infra)

These runs are **not TACC-specific**. The `.sbatch` files are ordinary bash —
the `#SBATCH` lines are comments and are ignored off-SLURM. On any machine with
the `.sif` (or native FIDESlib) + an A100-class GPU, run a block directly **from
the repo root**:

```bash
bash kimon/configs/config1_simd/block.sbatch     # apptainer mode (default)
RUN_MODE=native bash kimon/configs/config3_all_opts/block.sbatch   # native FIDESlib
```

Pin to a specific GPU with `CUDA_VISIBLE_DEVICES=<id>`. The one-block runs are
the reusable primitive — the same binary/fixture composes to end-to-end (configs
1 & 3 do exactly that across 12 blocks + head).

---

## 4. Get the results back

Everything lands under `kimon/logs/<config>/<tag>/`:
- `<tag>.json` — the result (timings, accuracy `global_rel_inf`, `passed`).
- `run.log` (or `reader_*.log` for sharding) — full output, ends with `[PASS]`/`[FAIL]`.

Copy the whole logs folder to your laptop / to us:
```bash
rsync -av <tacc-host>:<repo>/kimon/logs/ ./kimon/logs/
```

### Derive the speed improvements

Once you have some result JSONs, one command turns them into a comparison table
+ derived speedups (config1-vs-config3 diagcache benefit, block→e2e scaling,
config2's 2-GPU projection concurrency):
```bash
python3 kimon/analyze/summarize.py            # prints + writes kimon/logs/summary.md
python3 kimon/analyze/summarize.py --json     # also writes kimon/logs/summary.json
```
It only needs Python 3 (no packages) and reads whatever runs exist — run it on
TACC or after rsync'ing the logs back. `summary.md` is what we'll use to read
off the speedups; send it along with the logs.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `module: command not found` / `tacc-apptainer` not found | try `module load apptainer`; on non-TACC just ensure `apptainer` is on PATH |
| `.sif not found` when submitting | run Step A (`build_sif.sh`) first |
| `[FATAL] fixture missing` | run `kimon/env/prepare_fixtures.sh`; follow its scp hint |
| `[FATAL] <binary> missing` | run Step C (`build_binaries.sh`) inside a GPU node |
| job dies instantly, `-A` error | you didn't set `#SBATCH -A <allocation>` in the .sbatch |
| `partition ... not available` | fix `#SBATCH -p` to your system's A100 partition |
| `FIDESlib commit … expected …` | the .sif was built from the wrong image; rebuild from the correct `fideslib.tar` |
| runs but `[FAIL]` | check `run.log`; if `global_rel_inf` > 4e-2 tell us (accuracy issue), else it's an env problem |

---

## Reference (not needed to run)

**What runs:**

| Config | one block (T=103) | end-to-end (12 blocks + GSR head) |
|---|---|---|
| **1. SIMD batching** | ✅ `block.sbatch` | ✅ `e2e.sbatch` (15-45 h) |
| **2. GPU sharding** (Q/K/V projection, 2 GPU) | ✅ `block.sbatch` (projection step only) | ❌ not feasible — see below |
| **3. All optimizations** (SIMD+depth13+diagcache) | ✅ `block.sbatch` | ✅ `e2e.sbatch` (15-45 h) |

**How the TACC port works:** `env/common.sh` runs our existing `run_*.sh`
scripts inside the .sif via `apptainer exec --nv --bind <repo>:/work`; binaries
get `--gpu 0/1` and SLURM controls which physical GPUs are visible (no docker
`--gpus device=N` remap). Sharding's secret-key state dir is deleted on exit.
Image is `sm_80` (A100); H100 would need a rebuild (`FIDESLIB_ARCH=90`).

**Known limit — Config 2 is projection-only, and its e2e is not feasible.** The
2 GPUs each compute part of the attention Q/K/V *projection*, then stop.
Continuing (attention/MLP) or going end-to-end would require passing a
**ciphertext** between the two processes, which the pinned FIDESlib (786c7600)
cannot serialize (it serializes only the context/keys). This is the same wall
that blocks 2-GPU MLP sharding. Details: `docs/hybrid/tasks.md` (2026-08-01) and
`docs/hybrid/roadmap.md` item 12.
