# CLAUDE.md — project charter

Canonical instructions for every human or agent in this repository. Read this, then
`docs/overview.md` and `docs/tasks.md`.

## Objective

Determine whether **DNAGPT**
([TencentAILabHealthcare/DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT), arXiv 2307.05628)
inference can be executed under **Fully Homomorphic Encryption** — a compute provider evaluating the
model on an **encrypted genome** without receiving plaintext — and measure where correctness,
performance, or memory would prevent a complete encrypted deployment. A correct-but-slow encrypted
path is a valid outcome; feasibility and practicality are separate verdicts — never tune claims or
parameters to force "practical".

### Phase A (complete): plaintext baseline = the oracle

An encrypted run is only meaningful against the exact plaintext prediction it must reproduce. So we
first establish that DNAGPT is a good, reproducible model on three downstream tasks and freeze its
per-example predictions as the FHE acceptance oracle:

1. **Genomic Signal & Region Recognition (GSR)** — human signal-motif classification.
2. **GUE (Genome Understanding Evaluation)** — human promoter & splice-site classification.
3. **Human mRNA Abundance Regression.**

A task "passes" when DNAGPT's local metric is at/near the published reference; a negative boundary is
valid — never tune to force a pass.

### Phase B (current): encrypted-inference feasibility

The encrypted target starts at embedded numeric token vectors and ends at the released task head.
Every accepted result must match the Phase-A oracle within the declared tolerance. Encrypted
token-index lookup remains a separate unresolved boundary. See `docs/roadmap.md`.

**Default architecture: client-assisted CKKS** (legacy internal tag `Scheme B`, code in
`fhe/gpu_real_scheme_b/`). Server keeps all linear algebra (FIDESlib GPU, unchanged) in one
encrypted lineage; the client — the data owner, who already holds the secret key — decrypts only
ciphertexts derived from its own query at pre-declared nonlinearity boundaries (LayerNorm,
attention nonlinearity, GELU), evaluates exactly in plaintext, and re-encrypts. The untrusted
compute provider never observes plaintext, a partial decrypt, or the secret key. This is the only
architecture with active/planned work.

`[V]` The active implementation closes one complete real-weight block at the 103-token GSR prompt
length with eight-token SIMD packing and depth 13. `[V]` Two released blocks compose at two tokens
through a declared client refresh. `[V]` The complete 12-block plus GSR-head driver has now run and
passed at 103 tokens (2026-08-12, TACC Lonestar6): `6683 s` (~1.86 h) wall clock on a whole-node
allocation with no contention detected in recorded telemetry, final label matching the frozen
Phase-A oracle — see `docs/hybrid/tasks.md`. This is one sample, not yet reproduced a second time;
earlier long timings on other (non-T123-optimized or contended) runs remain
contaminated/superseded as documented there. Remaining performance work is
reproducing this result for variance, further exact-model optimization gates in `docs/hybrid/roadmap.md`,
and a real networked client/server transport measurement — not an unprofiled multi-day run.

Two alternatives were evaluated and are **not** the default; full justification in
`docs/shared/architecture_options.md`:

- **Pure non-interactive CKKS** (internal tag `Scheme A`, frozen baseline/ablation): one
  uninterrupted ciphertext lineage, zero intermediate decrypt. Closed one full real-weight block,
  then hit a root-caused GPU memory wall at chained multi-block composition. Kept only as the
  paper's ablation; no further Scheme A runs planned.
- **CKKS↔FHEW scheme switching** (internal tag `Scheme C`, deferred): fully non-interactive,
  exact nonlinearities via LUT, but no GPU-accelerated implementation exists to build on. Not
  rejected, just deferred.

Concrete ML's TFHE-rs backend was also evaluated and rejected (weak GPU speedup, large
ciphertext expansion) — see `docs/shared/backend_selection.md`.

**In scope:** the DNAGPT model graph as the FHE target; plaintext baseline harnesses; dataset
provenance; measured metrics; the evidence trail feeding the paper.
**Out of scope (for now):** wet-lab/clinical claims; encrypted token-index embedding
lookup; a production client/server key-custody and transport service.

## Locked decisions

- **Weights:** 0.1b only (`dna_gpt0.1b_h`, `dna_gpt0.1b_m`, released `classification.pth`,
  `regression.pth`). 3b deferred.
- **Device:** Phase A uses local Mac MPS/CPU float32. Phase B uses Mac Docker
  emulation for primitive accuracy, native Brev CPU for the complete toy-block anchor,
  then Brev A100 through the C++/CUDA FIDESlib performance path.
- **Task heads:** GSR + mRNA use the *released fine-tuned* heads (inference only). GUE has no
  released head → we fine-tune the foundation backbone ourselves (`eval/finetune_gue.py`).
- **Metrics:** GSR = accuracy/F1; mRNA = r² + Pearson/Spearman; GUE = MCC (primary) + acc/F1.

## Hard rules

1. **Never commit weights or datasets.** `checkpoints/`, `data/`, and large derived artifacts are
   gitignored. They are re-downloadable; the repo stays code + docs + small evidence JSON.
2. **Every dataset gets provenance** in `docs/data_provenance.md`: exact source URL, retrieval date,
   any recovery route (e.g. Internet Archive), preprocessing, and license. No silent data.
3. **Every result is an immutable run** under `results/runs/<tag>.json` (+ `_preds.csv` when
   applicable) and one row in the appropriate `results/{pure,hybrid,shared}/manifest.yaml`.
   Never rewrite an old run; add a new tag.
4. **Separate measured facts from claims.** Tag load-bearing statements `[V]` (verified/measured),
   `[U]` (unresolved/blocked), or `[A]` (assumption) with a source.
5. **Reproducibility:** every reported number has a single exact command in `docs/tasks.md` that
   regenerates it from the pinned `.venv`.
6. Read before editing; keep diffs reviewable; do not commit/push/deploy unless asked.

## The paper

`paper-docs/` holds the manuscript and everything feeding it. The manuscript answers the
**Objective** above directly — whether DNAGPT inference can run under FHE, and where correctness,
performance, or memory would stop a complete encrypted deployment — and it reports **feasibility
and practicality as separate verdicts**, exactly as this charter requires. The measured unit is one
complete transformer block at the task's own 103-token prompt length; whole-model figures are
labelled projections at fixed circuit.

Rules for writing it are in `paper-docs/AGENTS.md`. Every number it may print lives in
`paper-docs/evidence/*.yaml` with a source and a `[V]`/`[U]`/`[A]` tag; `paper-docs/scripts/`
enforces that mechanically, along with a ban on internal shorthand (`Scheme A`/`Scheme B`, run
tags, host names) reaching a reader. Timing measured under host contention does not enter the
manuscript in any form.

## Sources of truth

| Concern | File |
|---|---|
| What/why/status | `docs/overview.md` |
| Per-task method + commands + verdicts | `docs/tasks.md` |
| Evaluation methodology & design (why these metrics, harness design, oracle) | `docs/eval_approach.md` |
| Dataset origins, recovery, licenses | `docs/data_provenance.md` |
| Cross-project execution order | `docs/roadmap.md` |
| Phase-B rationale shared by both schemes (backend choice, architecture comparison) | `docs/shared/` |
| Phase-B Scheme A (pure, frozen) evidence | `docs/pure/` |
| Active client-assisted CKKS roadmap and evidence | `docs/hybrid/roadmap.md`, `docs/hybrid/tasks.md` |
| Run provenance | `results/{pure,hybrid,shared}/manifest.yaml`, `results/runs/` |
| Evidence acceptance rules | `results/README.md` |
| **The paper** — manuscript, figures, claims, writing rules | `paper-docs/README.md`, `paper-docs/AGENTS.md` |
| Numbers the paper may print | `paper-docs/evidence/*.yaml` |

## Repository layout

```text
DNAGPT/            cloned upstream model code (unmodified)
checkpoints/       downloaded 0.1b weights (gitignored)
data/{gsr,mrna,gue}/  datasets (gitignored; see docs/data_provenance.md)
eval/              common.py + eval_gsr.py + eval_mrna.py + build_mrna_testset.py + finetune_gue.py
fhe/               OpenFHE oracle plus FIDESlib CUDA toy/real-width gates
docker/            pinned OpenFHE Python and patched FIDESlib CUDA environments
results/           runs/ (immutable evidence) + README.md + pure/, hybrid/, shared/ manifests
docs/              overview, tasks, eval_approach, data_provenance, roadmap + pure/, hybrid/, shared/
paper-docs/        the paper: context/, evidence/, manuscript/, scripts/, reviews/, submission/
requirements.txt   pinned Phase-A dependencies
```

## GPU execution

Historical launchers under `fhe/gpu_real_scheme_b/` remain for result reproduction; the generic
`wait_and_run_scheme_b.sh` workflow targets gates that are already complete and is not the current
queue. Before any new remote run:

1. identify the exact gate in `docs/hybrid/roadmap.md`;
2. run the local contract suite;
3. use the version-matched launcher documented in `docs/hybrid/brev_runbook.md` and
   `docs/hybrid/tasks.md`;
4. require a dedicated or otherwise measured-clean host for performance claims; and
5. create new evidence rather than overwriting an old run.

The 12-block driver has closed arithmetic correctness once at task length. Treat its `6683 s` result
as a single-sample feasibility measurement, not a stable latency target. Before any optimization or
service-rate claim, obtain a current T=103 profile, run controlled paired gates, and repeat the
complete execution after retained changes are integrated.

## Base validation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# sanity: torch + MPS + the upstream package resolve
python -c "import sys; sys.path.insert(0,'DNAGPT'); import torch, dna_gpt; print('mps', torch.backends.mps.is_available())"
```

Data + weights are not committed — see `docs/data_provenance.md` (datasets) and the Google-Drive
folder `10UPPx6V13oQW6knuLV7d8SRIA3D6hYor` (0.1b weights) to rehydrate before running.

## Definition of done (per task)

- Data sourced with full provenance; preprocessing scripted (no manual steps).
- Metric measured on the declared evaluation set via one documented command, with held-out versus
  all-example scope stated explicitly.
- Immutable run JSON + manifest row written.
- Verdict recorded in `docs/tasks.md` with `[V]`/`[U]` vs the published reference.
