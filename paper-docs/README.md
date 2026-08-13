# paper-docs

Everything for the paper on client-assisted homomorphic inference for DNAGPT. The rest of the
repository is the research record; this directory is the publication.

Working here? Read [`AGENTS.md`](AGENTS.md) first — it is the contract, and it applies to people
as well as agents.

## Picking this up for the first time

```bash
brew install tectonic                  # or use an existing latexmk
paper-docs/scripts/build.sh            # figures, PDF, lint
open paper-docs/manuscript/source/main.pdf
```

Then read, in this order: this file's [Status](#status) section for what is and is not done,
[`AGENTS.md`](AGENTS.md) for the five rules that govern every edit, and
[`context/00_terminology.md`](context/00_terminology.md) for the vocabulary the lint enforces.

Two things worth knowing before you change anything. Numbers never go straight into the LaTeX —
they go into `evidence/*.yaml` with a source and a `[V]`/`[A]`/`[U]` tag, and the lint fails on any
number in the manuscript that is not there. And `[V]` measured, `[A]` derived or projected, and
`[U]` not measured are load-bearing distinctions throughout, not stylistic ones.

## Layout

| Directory | What it holds |
|---|---|
| [`context/`](context/) | what the paper may say and how — terminology, the deployment scenario, the research narrative, defensible claims |
| [`evidence/`](evidence/) | every number the manuscript may print, with its source and evidence tag. The only numeric authority |
| [`manuscript/`](manuscript/) | `source/` (LaTeX) and `figures/` (generated PDFs, committed) |
| [`scripts/`](scripts/) | figure generation, the number and terminology lint, the build wrapper |
| [`reviews/`](reviews/) | dated agent review output, appended never overwritten |
| `submission/` | packaged submission. Build artifact, never hand-edited. Not created yet |

## Build

```bash
paper-docs/scripts/build.sh            # figures, PDF, lint
paper-docs/scripts/build.sh figures    # figures only
paper-docs/scripts/build.sh lint       # numbers, terminology, citations
```

First run creates `paper-docs/.venv-paper` and installs `requirements.txt`. The repository's root
`.venv` is pinned to reproduce the plaintext baseline and must not gain dependencies.

A LaTeX engine is needed for the PDF step. The build uses `latexmk` when it is present and falls
back to `tectonic`, which is a single binary that fetches the packages it needs on first run:

```bash
brew install tectonic                  # macOS, small and self-contained
brew install --cask mactex-no-gui      # macOS, full TeX Live
apt install texlive-full latexmk       # Debian
```

The compiled `manuscript/source/main.pdf` is a build artifact and is not committed.

## How the pieces connect

```
evidence/*.yaml ──▶ scripts/figures.py ──▶ manuscript/figures/*.pdf ──┐
      │                                                              ├──▶ main.pdf
      └──────────▶ scripts/check_numbers.py ◀── manuscript/source/*.tex ┘
```

Figures and tables read the same ledger, so they cannot disagree. The lint fails on a number in
the LaTeX that is absent from the ledger, on internal shorthand that should not reach a reader,
and on unverified or dangling citations.

## The five agents

Defined in `.claude/agents/`. Invoke them rather than doing their work inline.

| Agent | Role | Edits |
|---|---|---|
| `paper-structure` | argument, section order, whether the evidence licenses the claims | yes |
| `paper-figures` | designs and generates figures from the ledger | yes |
| `paper-style` | academic voice, anti-machine-prose, mechanical consistency | yes |
| `paper-evidence` | traces every number and scope word to its source | no |
| `paper-sources` | verifies every citation exists and supports its sentence | no |

The two verifiers cannot edit by design.

Order of work: evidence lands in the ledger, `paper-structure` places it, prose gets drafted,
`paper-figures` builds anything visual, the two verifiers check, and `paper-style` polishes last.

## What the paper argues

It answers the project objective directly: **can DNAGPT inference run under homomorphic
encryption, and where would correctness, performance, or memory stop a complete encrypted
deployment?** The arc is:

1. establish the target is a genuine genomic language model, and freeze its predictions (§4);
2. build and evaluate two CKKS designs, reporting where each one stops (§5);
3. show what engineering moved, with the encrypted operation schedule held invariant (§6);
4. report **feasibility** and **practicality** as two separate verdicts (§7).

Correctness is not the barrier. Memory is a barrier that design moves. Cost is the remaining
constraint and it responds to engineering. A correct-but-slow encrypted path is the honest answer
and is reported as one.

## Status

Last updated 2026-08-08. **The paper is a complete, compiling draft, ready to be read end to end.**

**Done:**

- All ten sections drafted. No drafting placeholders remain in the body.
- The evidence ledger holds 119 tagged rows — 93 `[V]`, 22 `[A]`, 4 `[U]`. Every number in the
  LaTeX resolves to one of them.
- Nine figures generated from the ledger and placed in the text.
- Four protocol algorithms in §5, transcribed from the implementation and cross-checked against
  `context/08_implementation_ground_truth.md`, which carries the `file:line` citations.
- All 22 `refs.bib` entries verified against the live record (`reviews/2026-08-08-paper-sources.md`).
  Every entry is now cited.
- `scripts/build.sh` runs figures, PDF, and lint. All four lint checks clean; the PDF compiles with
  no undefined citation or cross-reference.

**Open, in rough priority order:**

1. **Three of the five review agents have never run.** Only `paper-sources` has a report in
   `reviews/`. `paper-evidence`, `paper-structure`, and `paper-style` are outstanding, in that
   order — structure before style, since polishing prose that may be restructured is wasted work.
2. **The twelve-block evaluation has not been run** (`compose.twelve_block_at_task_length`, `[U]`).
   Table 4 and the scaling figure are shaped to absorb it as one row and one point without
   disturbing the argument. §7 states the gap plainly today.
3. **Two smaller `[U]` items:** `client.share_latest` in `evidence/measurements.yaml` and
   `open.mask_plaintext_encoding` in `evidence/optimizations.yaml`.
4. **Confirm the pre-optimization baseline block time and the client-side figures** — see
   `open_provenance` in `evidence/optimizations.yaml`.
5. **Author metadata:** corresponding-author address in `sections/00_frontmatter.tex`, and the
   acknowledgements and compute-allocation statements in `sections/10_conclusion.tex`. These are
   the only `TODO` markers left in the manuscript.

**Recently corrected — worth knowing if you read an earlier draft.** The genomic-signal baseline
reference was attributed to DeepGSR at 0.916. It is in fact DNAGPT's own reported figure for this
model class, and it comes from a held-out quarter of the set rather than the full 22,604 examples
this work evaluates. The ledger, Table 1, and the §4 prose now say so. DeepGSR's own reported
accuracy for the task is 0.8694.
