# Overview

## Purpose

Evaluate whether **DNAGPT** (arXiv 2307.05628) inference can run under **FHE** so a compute
provider never sees plaintext DNA. Two architectures are tracked as of 2026-07-25
(`docs/feasibility/05_architecture_options.md`): **Scheme A**, pure non-interactive CKKS with zero
intermediate decryption, now frozen as the paper's baseline/ablation after closing one full block
and then hitting a root-caused GPU memory wall at chained composition; and **Scheme B**, the active
path — hybrid client-assisted CKKS, where server-side linear algebra stays encrypted end to end and
only the data-owning client (already the secret-key holder) decrypts at pre-declared nonlinearity
boundaries. Encrypted token-index embedding lookup remains a separate `[U]` boundary for both
schemes.

**Phase A is complete:** DNAGPT passes three local downstream-task gates and its
per-example predictions are frozen as plaintext oracles. **Phase B is active:** toy
arithmetic and CUDA backend parity are complete under Scheme A; work is now shifting to
Scheme B's hybrid block gate, then sequence scaling (see [roadmap.md](roadmap.md)).

## What DNAGPT ships

Inference only: `test.py` + `dna_gpt/`. Released fine-tuned heads (Google Drive, 0.1b):
`classification.pth` (GSR, on `dna_gpt0.1b_m`) and `regression.pth` (mRNA, on `dna_gpt0.1b_h`,
max_len 4096). No datasets, no fine-tuning code, no GUE head — we supply those.

## Status

| # | Task | Verdict | Result (local) | Reference |
|---|------|---------|----------------|-----------|
| 1 | GSR — human AATAAA PAS | **[V] PASS** | acc **0.9124**, F1 **0.916** (n=22,604) | DeepGSR ~0.916 |
| 3 | mRNA abundance regression | **[V] PASS** | r² **0.562**, Pearson **0.753**, Spearman 0.745 (n=1,000) | DNAGPT paper r² ~0.62 |
| 2 | GUE — promoters & splice sites | **[V] PASS** | MCC prom_core **0.680** / prom_300 **0.897** / splice **0.831** | DNABERT-2 ~0.69/0.87/0.85 |

**Phase A complete: all three tasks pass locally.** Per-example predictions in `results/runs/*_preds.csv`
are the frozen oracle for the encrypted (FHE) path.

See [tasks.md](tasks.md) for methods/commands and [data_provenance.md](data_provenance.md) for sources.

## Phase B status

- `[V]` OpenFHE CKKS linear, LayerNorm, causal-softmax, GELU, and bootstrap
  primitives pass at `HEStd_128_classic`.
- `[V]` A complete `D=8`, `T=4`, two-head block passes natively on Brev CPU
  with global rel-inf `1.49e-3`, zero intermediate decrypts, and one final decrypt.
- `[V]` The complete toy graph passes the C++/CUDA A100 gate. Released-weight
  `D=768`, `T=2` LayerNorm and 12-head attention/projection gates also pass.
- `[V]` The original full-block depth-43 schedule fails closed from deterministic
  level exhaustion before any final decrypt; the exact T=2 sigmoid reformulation
  is the active replacement.
- `[V/A]` The fixed plaintext nonlinear schedule passes all 12 released blocks
  and the GSR head on the public T=2 fixture.
- `[V]` Local probes validate BSGS rotation reduction, numerator-first attention, and
  lower-degree GELU candidates without changing the correctness gate.
- `[U]` The optimized real-width MLP/full block, encrypted refresh/composition,
  broader public calibration, and a task-valid encrypted sequence remain under Scheme A;
  Scheme A is now frozen as the paper's non-interactive baseline/ablation.
- `[V]` Three independent chained-composition attempts (`multiplicative_depth` in
  `{50, 58, 64}`, plus a 2-GPU variant) all fail closed on GPU memory exhaustion during
  rotation-key/bootstrap-plaintext loading, not accuracy — root-caused via a symbolized
  backtrace. This forced a pivot to **Scheme B** (hybrid client-assisted CKKS): see
  [feasibility/05_architecture_options.md](feasibility/05_architecture_options.md).

The full twelve-layer CPU route is not a planned stage: after a complete block closes,
it adds no new arithmetic claim. Performance work moves to C++/CUDA with current
FIDESlib/OpenFHE interoperability. The ordered gates and skip rules are documented in
[roadmap.md](roadmap.md).

## Environment

Mac (darwin arm64), Python 3.12, torch 2.13 (MPS), float32. 0.1b models run comfortably on MPS
(GSR ~6 ms/seq, mRNA ~67 ms/seq at 10.5 kb, GUE fine-tune ~minutes/epoch).

## Key provenance note

The Xpresso mRNA dataset host (`krishna.gs.washington.edu`) is **globally offline**; the canonical
human input was recovered from the **Internet Archive** and reprocessed with a faithful Python-3 port
of Xpresso's split (18,377 genes → last 1,000 = test). Details in [data_provenance.md](data_provenance.md).
