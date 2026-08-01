# Encrypted DNAGPT inference

This repository evaluates whether the released 0.1-billion-parameter
[DNAGPT](https://github.com/TencentAILabHealthcare/DNAGPT) model can process encrypted embedded DNA
tokens without exposing plaintext activations to an untrusted compute provider.

Correctness and practicality are separate questions. A slow but correct encrypted path is a valid
feasibility result; contaminated timing is not a performance benchmark.

## Current result

| Stage | State |
|---|---|
| Plaintext model validation | `[V]` Three task families pass and provide frozen numerical oracles |
| Pure non-interactive CKKS | `[V]` One real-weight block passes; frozen after chained composition exceeded the tested GPU memory envelope |
| Client-assisted CKKS, one block | `[V]` Complete real-weight block passes at the 103-token GSR length with about `4e-9` relative error |
| Short composition | `[V]` Two released blocks pass at two tokens using a declared client refresh |
| Complete encrypted classifier | `[U]` The 12-block plus GSR-head driver is built but has not run at 103 tokens |
| Performance | `[U]` Existing long runs are shared-host contaminated; clean and networked end-to-end latency are unmeasured |

The active protocol keeps model projections, attention algebra, residuals, and the MLP encrypted on
the GPU server. At fixed nonlinear boundaries, the data-owning client decrypts its own intermediate,
computes the exact nonlinearity, and re-encrypts it. The server does not receive plaintext or the
secret key under this prototype boundary.

The immediate performance decision is to profile and optimize the current task-length block before
using the existing multi-day driver as a final latency experiment. See the
[active execution roadmap](docs/hybrid/roadmap.md).

## Start here

- [Project charter and rules](CLAUDE.md)
- [Current overview](docs/overview.md)
- [Cross-project roadmap](docs/roadmap.md)
- [Client-assisted CKKS roadmap](docs/hybrid/roadmap.md)
- [Paper evidence sourcebook](docs/paper/README.md)
- [Dataset provenance](docs/data_provenance.md)

## Repository layout

```text
DNAGPT/              upstream model code
eval/                plaintext evaluation and fine-tuning harnesses
fhe/                 encrypted arithmetic, GPU implementations, and contracts
results/             immutable run evidence and manifests
docs/                canonical status, methods, roadmaps, histories, and paper notes
docker/              pinned OpenFHE/FIDESlib environments
data/, checkpoints/  re-downloadable, gitignored inputs
```

## Reproduce the plaintext baselines

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eval/eval_gsr.py --limit -1 --tag gsr_aataaa_human_full
python eval/build_mrna_testset.py
python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all --max_len 64
```

Weights and datasets are intentionally absent. Rehydration and provenance are documented in
[docs/data_provenance.md](docs/data_provenance.md); exact accepted commands and results live in
[docs/tasks.md](docs/tasks.md).
