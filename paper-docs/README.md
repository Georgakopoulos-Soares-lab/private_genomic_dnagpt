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
| [`manuscript/`](manuscript/) | `source/` (LaTeX), `figures/` (generated PDFs, committed), and `dnagpt-fhe-paper.pdf`, the committed snapshot of the compiled manuscript |
| [`scripts/`](scripts/) | figure generation, the number and terminology lint, the build wrapper |
| [`reviews/`](reviews/) | dated agent review output, appended never overwritten |
| `submission/` | packaged submission. Build artifact, never hand-edited. Not created yet |

## Build

```bash
paper-docs/scripts/build.sh            # figures, PDF, lint — preprint (no line numbers)
paper-docs/scripts/build.sh --review   # same, with line numbers for referees
paper-docs/scripts/build.sh figures    # figures only
paper-docs/scripts/build.sh lint       # numbers, terminology, citations
```

The default build is the preprint. `--review` turns on `lineno` margin numbers so referees can cite
a page and line; the script writes `source/buildmode.tex` on every run, so a stale flag can never
leak line numbers into a preprint or drop them from a reviewer copy. The committed snapshot is a
preprint build.

First run creates `paper-docs/.venv-paper` and installs `requirements.txt`. The repository's root
`.venv` is pinned to reproduce the plaintext baseline and must not gain dependencies.

A LaTeX engine is needed for the PDF step. The build uses `latexmk` when it is present and falls
back to `tectonic`, which is a single binary that fetches the packages it needs on first run:

```bash
brew install tectonic                  # macOS, small and self-contained
brew install --cask mactex-no-gui      # macOS, full TeX Live
apt install texlive-full latexmk       # Debian
```

The compiled `manuscript/source/main.pdf` is a build artifact and is not committed: it is rewritten
on every build, so tracking it would churn the repository.

A snapshot of the current compiled manuscript is committed separately at
[`manuscript/dnagpt-fhe-paper.pdf`](manuscript/dnagpt-fhe-paper.pdf) so the paper can be read
without a LaTeX toolchain. It is a copy, not a build output — refresh it deliberately after a clean
build whose four lint checks pass:

```bash
paper-docs/scripts/build.sh
cp paper-docs/manuscript/source/main.pdf paper-docs/manuscript/dnagpt-fhe-paper.pdf
```

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

Last updated 2026-08-14. **The paper is a revised, compiling draft reporting a measured complete
encrypted inference.** All four lint checks are clean; seven retained figures are generated from
the ledger and placed in the text.

### What the paper now reports

The complete model — all twelve released blocks and the released task head — was executed at the
103-token task prompt on 2026-08-12 on a whole-node A100 allocation with no contention detected in
recorded telemetry. It reproduces the frozen plaintext label with `8.56e-09` relative margin error
against the unchanged `4e-2` tolerance, in `6683 s` (1.86 h), with no homomorphic bootstrap.
Device-wide GPU memory used during the run reaches 9839 MiB after about 400 s and remains flat for
the final 94% of the execution.
That run is banked in `evidence/measurements.yaml` under `complete_model`, with its contention
check in `evidence/telemetry.yaml` under `complete_model_contention`.

This changed the paper's headline. Feasibility is now affirmative for a complete model rather than
for one block plus a composition argument, and practicality has a real negative verdict at a
measured cost instead of being withheld.

### Known limits of the current evidence — read before strengthening any claim

These are not open work items; they are the boundaries the current draft is written to respect.
Weakening them requires new measurements, not new wording.

- **The complete-model run is a single execution.** Every timing in the paper is `n = 1`. The text
  says so in §7, §8, and the conclusion. Do not convert it into a mean, a rate, or a latency
  target, and do not drop the qualifier to make a sentence read better.
- **Per-block timings are not reported at all.** The per-block numbers still in
  `evidence/measurements.yaml` under `block_timing` trace to a shared-partition run. They stay in
  the ledger because the ledger is the project's memory, but they must not reach the manuscript.
- **The optimization campaign carries no speedup figure.** The `11.6x` factor in
  `evidence/optimizations.yaml` needs a dedicated-node measurement of the pre-optimization
  baseline, which does not exist. §6 therefore reports what changed and what was held invariant,
  not how much faster it got. There is no waterfall figure for the same reason; its generator is
  retained in `scripts/figures.py` but unregistered.
- **`client.peak_ram` is attested, not committed.** It is absent from the manuscript and from the
  graphical abstract.
- **Sequence-length scaling is circuit size, not time.** `evidence/scaling.yaml` carries ciphertext
  groups and causal score tiles per task. The former per-task time projections were built on the
  shared-partition block time and have been removed.
- **Two `[U]` rows remain:** `client.share_latest` and `open.mask_plaintext_encoding`.

### Author inputs before submission

1. Add the corresponding-author email address or addresses; the draft currently names the
   correspondents without inventing contact details.
2. Add biographies or acknowledgements only if the target venue requires them. Empty placeholders
   and `TODO` markers have been removed from the manuscript.
3. Select and record a repository reuse license. The code-availability statement currently says,
   accurately, that no repository-level license has been assigned.

### Completed local checks

- The complete PDF has been rebuilt and all 25 pages visually inspected.
- All seven retained figures fit at manuscript size; the misleading cross-study baseline chart was
  removed.
- Algorithms 1--4 appear before the references, all fonts are embedded, and no clipping, overlap,
  missing glyph, or unresolved-reference marker was found.
- Terminology, number, citation, and structure lint all pass. Independent evidence and source
  closure checks also pass.
