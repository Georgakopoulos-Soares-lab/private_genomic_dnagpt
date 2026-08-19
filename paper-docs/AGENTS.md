# AGENTS.md — working on the paper

Contract for any human or agent writing, editing, or reviewing the manuscript. It overrides the
repository's general engineering guidance for anything under `paper-docs/`.

## Read first, in this order

1. [`context/00_terminology.md`](context/00_terminology.md) — banned terms, terms of art, claim
   discipline. Non-negotiable and mechanically enforced.
2. [`context/01_scenario_and_motivation.md`](context/01_scenario_and_motivation.md) — the
   deployment scenario the whole paper argues from.
3. [`evidence/README.md`](evidence/README.md) — where numbers come from.
4. [`context/07_source_map.md`](context/07_source_map.md) — which repository file answers which
   manuscript question.
5. [`context/08_implementation_ground_truth.md`](context/08_implementation_ground_truth.md) — what
   the system actually does, verified against the source with `file:line` citations. It carries the
   protocol pseudocode, the re-derivation of every operation count, and a claims audit. Where it
   and any other document disagree about mechanism, it wins.

Do not reconstruct the state of the project from filenames, run tags, or commit messages. The
context notes and the evidence ledger own it.

## The five rules

**1. Numbers come from the ledger.** Every numeric value in `manuscript/` must exist in
`evidence/*.yaml`. If you need a number that is not there, add it to the ledger with its source
and evidence tag first, then use it. Never type a number straight from a document into a `.tex`
file.

**2. The paper answers a feasibility question.** From the project objective: *can DNAGPT
inference be executed under homomorphic encryption, and where would correctness, performance, or
memory prevent a complete encrypted deployment?* Everything in the manuscript serves that
question. Feasibility and practicality are **separate verdicts**, reported separately, in
§\ref{sec:verdict-feasibility} and §\ref{sec:verdict-practicality}. A correct-but-slow
encrypted path is a real answer, not a failure — never tune a claim or a parameter to reach
"practical".

State the measured scope once, where it belongs. One complete transformer block at the task's own
prompt length remains the unit for block-level operation counts and stage memory. The complete
twelve-block model plus task head has also executed once at that prompt length, so its timing,
memory, and label agreement are measured single-sample results rather than projections. Do not turn
that one execution into a mean, variance, service rate, encrypted task accuracy, or input-domain
correctness claim, and do not repeat the same limitation in every summary layer.

**3. Measured, projected, and assumed are different words.** `[V]` measured, `[A]` derived or
projected, `[U]` not measured. An operation-count reduction is never a speedup. Twelve times a
block time is never a latency. Say which one you mean, in the sentence where the number appears.

**4. No internal shorthand.** No `Scheme A`/`Scheme B`, no `T123`, no `v0`/`v1`/`v2`/`v3`, no
host names, no scheduler job numbers, no file hashes, no fixture or gate names, no engineer names
in the body. See the replacement table in `00_terminology.md`.

**5. Lead with what worked.** This is a paper about a complete released model that executed under
the declared protocol, with a fixed checked circuit and a systems design that bounds host and GPU
memory. The historical speedup factor lacks a controlled dedicated-node baseline and does not enter
the manuscript.

A failed attempt earns a place only when it explains a design decision the reader would otherwise
find arbitrary — chiefly why weight plaintexts are *cloned* rather than reused, and why client
boundaries run sequentially. One or two sentences each, inside the subsection they explain, framed
as the reason the retained design looks the way it does. No "what did not work" section, no
catalogue, no tally.

Timing measured under host contention does not appear in the manuscript at all — not as a result,
not as a caveat, not as a footnote. The reported timings come from the dedicated-node runs, which
is the honest and the stronger choice. Contended runs remain in the repository for provenance and
are cited, if at all, only for quantities that contention cannot affect: relative error, operation
counts, memory footprint, and multiplicative depth.

`evidence/optimizations.yaml` records everything, including the abandoned attempts, because the
ledger is the project's memory. The manuscript draws a deliberate subset from it.

## Terminology quick reference

| Never | Instead |
|---|---|
| Scheme A / Scheme B | non-interactive CKKS / client-assisted CKKS |
| T123, v1, v2, v3, config1-3 | the named optimization |
| Brev, TACC, Lonestar6, SLURM job N | one NVIDIA A100 GPU node with 32 CPU cores |
| fixture, oracle hash, gate, manifest row | the frozen plaintext reference |
| round trip | client boundary crossing |
| "the server learns nothing" | "the server receives no plaintext activation and no secret key" |
| "DNAGPT runs under FHE" | "one DNAGPT block runs with encrypted server-side linear algebra and client-evaluated nonlinearities" |

## Layout

```
paper-docs/
├── context/      what the paper may say, and how (read-only during drafting)
├── evidence/     every number, with source and tag (the only numeric source)
├── manuscript/   source/ (LaTeX), figures/ (generated PDFs, committed), and the committed
│                 compiled snapshot dnagpt-fhe-paper.pdf
├── scripts/      figure generation, number and terminology lint, build
├── reviews/      dated agent review output
└── submission/   arXiv package (build artifact — never hand-edited)
```

## Build

```bash
paper-docs/scripts/build.sh          # figures, PDF, lint — creates the venv on first run
paper-docs/scripts/build.sh lint     # just the four checks
paper-docs/scripts/build.sh docx     # regenerate the review exchange copy
```

The PDF step uses `latexmk` when present and falls back to `tectonic`.

`docx` writes `manuscript/exchange/dnagpt-fhe-paper.docx`, the copy sent out for comment. It is a
build artifact, gitignored, and never hand-edited: comments come back as text and are applied to the
`.tex` sources, then the copy is regenerated. Tables and algorithms travel as placeholders
(`manuscript/mkplaceholders.py`) so that no checked number can be edited in a word processor, and
figures are rasterised first because an embedded PDF does not render there. Needs `pandoc` and
`pdftoppm` (`brew install pandoc poppler`); it stages a temporary copy and never touches
`manuscript/source/`.

The paper uses its own virtual environment. The repository's pinned `.venv` reproduces Phase-A
results and must not gain new dependencies.

## The five specialist agents

Defined in `.claude/agents/`. Use them; do not do their jobs inline.

| Agent | Does | Can edit |
|---|---|---|
| `paper-sources` | verifies every citation resolves to a real work that supports the sentence citing it | no — reports only |
| `paper-evidence` | traces every number and scope word back to the ledger and its source | no — reports only |
| `paper-structure` | narrative architecture, argument flow, section balance | yes |
| `paper-style` | academic voice, anti-machine-prose, mechanical consistency | yes |
| `paper-figures` | designs and generates figures from the ledger | yes |

The two verifiers cannot edit on purpose. A checker that fixes its own findings stops being a
check.

Each writes a dated report to `reviews/`, named `YYYY-MM-DD-<agent>.md`. Reports are appended,
never overwritten — the review history is part of the record.

## Order of work

Evidence before prose, prose before polish.

1. The claim exists in the ledger with a source and a tag.
2. `paper-structure` places it in the argument.
3. Prose is drafted.
4. `paper-figures` builds anything visual, from the ledger.
5. `paper-evidence` and `paper-sources` verify.
6. `paper-style` polishes last, because polishing prose that is about to be restructured is
   wasted work.

## Do not

- Do not commit weights, sequences, activations, keys, ciphertexts, or credentials.
- Do not hand-edit anything under `submission/` — it is generated.
- Do not add dependencies to the repository's root `requirements.txt`.
- Do not tune a number, soften a limitation, or drop a negative result to strengthen a claim.
- Do not invent a citation. If a work cannot be verified, it does not go in the bibliography.
