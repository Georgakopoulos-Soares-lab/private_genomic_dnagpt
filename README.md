# Encrypted DNAGPT — FHE feasibility

Determine whether **DNAGPT** ([TencentAILabHealthcare/DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT),
arXiv 2307.05628) inference can be run under **Fully Homomorphic Encryption** — i.e. a compute
provider evaluates the transformer on **encrypted embedded numeric genomic-token vectors** without
ever seeing plaintext — and measure where correctness, performance, or memory would stop a complete
encrypted deployment. Encrypted token-index embedding lookup is a separate unresolved boundary.

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
| **B. FHE feasibility** | encrypted operators → toy block → GPU real-width blocks | **active** |

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
fhe/             CKKS oracle/operators, toy + real-width CUDA graphs, refresh/range/scale gates
docker/          pinned OpenFHE Python and patched FIDESlib C++/CUDA environments
```

## Start here

- Charter & rules: [CLAUDE.md](CLAUDE.md)
- Status & framing: [docs/overview.md](docs/overview.md)
- Per-task method + commands + verdicts: [docs/tasks.md](docs/tasks.md)
- Dataset origins (incl. Internet-Archive recovery of the dead Xpresso host): [docs/data_provenance.md](docs/data_provenance.md)
- Path to encrypted DNAGPT: [docs/roadmap.md](docs/roadmap.md)
- Phase-B evidence and boundary: [docs/feasibility/00_overview.md](docs/feasibility/00_overview.md)

## Phase B status

OpenFHE CKKS BSGS linear, LayerNorm, causal-softmax, GELU, and bootstrap-refresh
primitives pass independently at 128-bit security. A complete `D=8`, `T=4`,
two-head block passes on both the native Brev CPU anchor and the C++/CUDA A100 path.
The GPU run has global rel-inf `1.8423e-4`, worst-token rel-inf `4.8125e-4`,
zero intermediate decrypts, one final decrypt, and `93.405 s` encrypted evaluation.

Released-weight `D=768`, `T=2` block-0 LayerNorm and 12-head
attention/projection gates pass on A100s at global rel-inf `3.8193e-10` and
`7.0183e-10`. The original full-block depth-43 schedule then failed closed before
the final decrypt: its exp-plus-reciprocal attention left token 1 three levels
short. The algebraically exact T=2 sigmoid replacement is the active full-block
path, followed by refresh and composition. Separately, a fixed plaintext
approximation preflight passes all 12 released blocks and the GSR head within
the `4e-2` gate. See the roadmap for boundaries and evidence.

## Reproduce Phase A

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch numpy gdown scikit-learn pandas h5py scipy tqdm
python eval/eval_gsr.py --limit -1 --tag gsr_aataaa_human_full         # Task 1
python eval/build_mrna_testset.py && python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest  # Task 3
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all --max_len 64  # Task 2
```
