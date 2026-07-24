# Encrypted DNAGPT — FHE feasibility

Determine whether **DNAGPT** ([TencentAILabHealthcare/DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT),
arXiv 2307.05628) inference can be run under **Fully Homomorphic Encryption** — i.e. a compute
provider evaluates the model on an **encrypted genome** without ever seeing plaintext — and measure
where correctness, performance, or memory would stop a complete encrypted deployment. Sibling
project `../evo2` asks the same question for Evo 2; this repo is its DNAGPT counterpart.

A correct-but-slow encrypted path is still a valid research outcome. Feasibility and practicality are
separate verdicts — measure the boundary honestly, never tune to force a "practical" conclusion.

## Why a plaintext baseline first

An encrypted run is only meaningful against a **plaintext oracle**: the exact prediction the
encrypted computation must reproduce within a declared tolerance. So **Phase A** establishes that
DNAGPT is a good, reproducible model on three downstream tasks, and freezes per-example predictions
as the acceptance oracle for the encrypted path.

| Phase | Goal | State |
|---|---|---|
| **A. Plaintext baseline (oracle)** | DNAGPT measured locally on 3 tasks | **3/3 PASS** ✅ |
| **B. FHE feasibility** | encrypted operators on the 0.1b backbone → encrypted end-to-end | not started |

### Phase A status (the oracle)

| # | Task | Verdict | Result | Reference |
|---|------|---------|--------|-----------|
| 1 | Genomic Signal & Region Recognition (human AATAAA) | **[V] PASS** | acc **0.9124**, F1 0.916 (n=22,604) | DeepGSR ~0.916 |
| 3 | Human mRNA Abundance Regression | **[V] PASS** | r² **0.562**, Pearson 0.753 (n=1,000) | DNAGPT paper ~0.62 |
| 2 | GUE — promoters & splice sites | **[V] PASS** | MCC 0.680 / 0.897 / 0.831 (core/300/splice) | DNABERT-2 ~0.69/0.87/0.85 |

## What DNAGPT ships

Inference only: `test.py` + `dna_gpt/`. Released 0.1b heads `classification.pth` (GSR) and
`regression.pth` (mRNA). No datasets, no fine-tuning code, no GUE head — supplied here.

## Layout

```
DNAGPT/          cloned upstream model (unmodified) — the FHE target graph
checkpoints/     0.1b weights (gitignored)
data/{gsr,mrna,gue}/  datasets (gitignored; see docs/data_provenance.md)
eval/            plaintext baseline harnesses (GSR, mRNA, GUE fine-tune)
results/         immutable evidence: manifest.yaml + runs/ (+ *_preds.csv = the oracle)
docs/            charter overview, tasks, data provenance, roadmap-to-FHE
```

## Start here

- Charter & rules: [CLAUDE.md](CLAUDE.md)
- Status & framing: [docs/overview.md](docs/overview.md)
- Per-task method + commands + verdicts: [docs/tasks.md](docs/tasks.md)
- Dataset origins (incl. Internet-Archive recovery of the dead Xpresso host): [docs/data_provenance.md](docs/data_provenance.md)
- Path to encrypted DNAGPT: [docs/roadmap.md](docs/roadmap.md)

## Reproduce Phase A

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch numpy gdown scikit-learn pandas h5py scipy tqdm
python eval/eval_gsr.py --limit -1 --tag gsr_aataaa_human_full         # Task 1
python eval/build_mrna_testset.py && python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest  # Task 3
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all --max_len 64  # Task 2
```
