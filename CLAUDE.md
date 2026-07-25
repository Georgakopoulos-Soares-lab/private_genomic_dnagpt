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

### Phase B (current): FHE feasibility

Encrypted operators on the 0.1b backbone → encrypted end-to-end on a task, matching the Phase-A
oracle within a declared tolerance. See `docs/roadmap.md`.

Two architectures are tracked, per
`docs/feasibility/05_architecture_options.md`:

- **Scheme A (frozen baseline):** pure non-interactive CKKS, one uninterrupted ciphertext
  lineage, zero intermediate decrypt. Proven to close one full real-weight block; proven to hit
  a root-caused GPU memory wall at chained multi-block composition. Kept as-is for the paper's
  ablation/baseline; no further Scheme A runs planned unless needed to re-establish the boundary.
- **Scheme B (active path):** hybrid client-assisted CKKS. Server keeps all linear algebra
  (FIDESlib GPU, unchanged) in one encrypted lineage; the client — the data owner, who already
  holds the secret key — decrypts only ciphertexts derived from its own query at pre-declared
  nonlinearity boundaries (LayerNorm, attention nonlinearity, GELU), evaluates exactly in
  plaintext, and re-encrypts. The untrusted compute provider never observes plaintext, a partial
  decrypt, or the secret key. See `docs/feasibility/05_architecture_options.md` for the full
  comparison against Scheme A and a deferred Scheme C (CKKS↔FHEW scheme switching).

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
   applicable) and one row in `results/manifest.yaml`. Never rewrite an old run; add a new tag.
4. **Separate measured facts from claims.** Tag load-bearing statements `[V]` (verified/measured),
   `[U]` (unresolved/blocked), or `[A]` (assumption) with a source.
5. **Reproducibility:** every reported number has a single exact command in `docs/tasks.md` that
   regenerates it from the pinned `.venv`.
6. Read before editing; keep diffs reviewable; do not commit/push/deploy unless asked.

## Sources of truth

| Concern | File |
|---|---|
| What/why/status | `docs/overview.md` |
| Per-task method + commands + verdicts | `docs/tasks.md` |
| Evaluation methodology & design (why these metrics, harness design, oracle) | `docs/eval_approach.md` |
| Dataset origins, recovery, licenses | `docs/data_provenance.md` |
| Next steps toward FHE | `docs/roadmap.md` |
| Phase-B backend, operator, block, and Brev evidence | `docs/feasibility/` |
| Run provenance | `results/manifest.yaml`, `results/runs/` |
| Evidence acceptance rules | `results/README.md` |

## Repository layout

```text
DNAGPT/            cloned upstream model code (unmodified)
checkpoints/       downloaded 0.1b weights (gitignored)
data/{gsr,mrna,gue}/  datasets (gitignored; see docs/data_provenance.md)
eval/              common.py + eval_gsr.py + eval_mrna.py + build_mrna_testset.py + finetune_gue.py
fhe/               OpenFHE oracle plus FIDESlib CUDA toy/real-width gates
docker/            pinned OpenFHE Python and patched FIDESlib CUDA environments
results/           manifest.yaml + runs/ (immutable evidence) + README.md
docs/              overview, tasks, eval_approach, data_provenance, roadmap
requirements.txt   pinned Phase-A dependencies
```

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
- Metric measured on the canonical test split via one documented command.
- Immutable run JSON + manifest row written.
- Verdict recorded in `docs/tasks.md` with `[V]`/`[U]` vs the published reference.
