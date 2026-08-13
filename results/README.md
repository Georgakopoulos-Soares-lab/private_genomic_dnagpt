# Evidence contract

`results/` is the immutable evidence store. A performance or correctness number is accepted only when
it has a run file, manifest entry, and interpretation in the owning task document.

## Rules

- **One run, one tag.** Each evaluation writes `runs/<tag>.json` and, where per-example outputs matter,
  `runs/<tag>_preds.csv`. Never edit or overwrite a completed run; a repetition gets a new tag.
- **One manifest row.** Add every accepted run to `pure/manifest.yaml`, `hybrid/manifest.yaml`, or
  `shared/manifest.yaml` according to its architecture. The legacy directory names are stable evidence
  namespaces, not preferred paper terminology.
- **One exact command.** Record the reproduction command and interpretation in `docs/tasks.md`,
  `docs/pure/tasks.md`, or `docs/hybrid/tasks.md`.
- **No evidence by chat or log alone.** Logs can diagnose failures but do not create an accepted metric
  unless the evidence contract for that result explicitly includes them.
- **Preserve provenance.** FHE runs record the security/context shape, source provenance, level and
  boundary counts, numerical oracle comparison, environment, and latency qualification needed to
  reproduce and interpret the run.
- **Separate result types.** A derived count is not a measurement. A shared-host observation is not a
  clean benchmark. A failed run can establish a useful boundary if its failure mechanism and scope are
  captured honestly.

Per-example Phase-A predictions are the frozen oracles for later encrypted evaluation. Do not expose
weights, datasets, plaintext sequences, activations, keys, or ciphertext payloads in evidence files.

## Manifest ownership

| Manifest | Owns |
|---|---|
| `shared/manifest.yaml` | Plaintext task baselines and architecture-independent scaffolding |
| `pure/manifest.yaml` | Pure non-interactive CKKS baseline and failure boundary |
| `hybrid/manifest.yaml` | Active client-assisted CKKS correctness, optimization, and composition evidence |

The manifests are the complete run index. This README deliberately does not duplicate an exhaustive
table because such tables become stale and compete with the immutable index.

## Current headline evidence

- `[V]` Phase A: GSR, mRNA abundance, and the selected GUE promoter/splice tasks pass and supply
  plaintext oracles.
- `[V]` Pure non-interactive CKKS: one complete real-weight block passes; chained composition reaches a
  root-caused GPU memory boundary.
- `[V]` Client-assisted CKKS: one complete real-weight block passes at the 103-token GSR prompt length
  with eight-token SIMD packing, depth 13, approximately `4e-9` relative error, and approximately
  `9.8 GiB` target-process peak GPU memory.
- `[V]` Short client-assisted composition: two released blocks pass at two tokens.
- `[V]` **Complete 12-block plus GSR-head run at T=103: PASS** (2026-08-12, TACC Lonestar6), `6683 s`
  (~1.86 h) wall clock on a clean single A100, label matches the frozen Phase-A oracle. See
  `hybrid/manifest.yaml` and `../docs/hybrid/tasks.md`.
- `[U]` Repeated/clean-variance confirmation of that run and a networked protocol measurement remain
  open.

Current interpretation and execution order live in:

- [`../docs/overview.md`](../docs/overview.md)
- [`../docs/roadmap.md`](../docs/roadmap.md)
- [`../docs/hybrid/roadmap.md`](../docs/hybrid/roadmap.md)
- [`../docs/paper/03_results_and_limits.md`](../docs/paper/03_results_and_limits.md)
