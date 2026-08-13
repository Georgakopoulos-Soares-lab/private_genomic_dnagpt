# Evidence ledger

Four YAML files hold every number the manuscript is allowed to print. The figure scripts read
them. The number lint reads them. Nothing else is a source.

| File | Holds |
|---|---|
| `measurements.yaml` | the measured encrypted block, correctness, memory, operation counts, crypto parameters, plaintext baseline |
| `optimizations.yaml` | the optimization campaign: what was kept, what was abandoned and why, what remains open |
| `scaling.yaml` | per-task token derivations, the measured anchor, and projections to the other tasks |
| `telemetry.yaml` | the resource trace of the measured run, and what it shows |

## Why a ledger instead of citing documents directly

Three reasons, all learned the hard way on the group's previous paper.

A figure and a table cannot disagree if both are generated from the same file. The previous
paper's review caught a default value stated three different ways across five locations, and a
rounding mismatch between a figure and its own caption.

A number's scope travels with it. `652.545 s` means nothing on its own; `652.545 s of encrypted
evaluation for one transformer block at 103 tokens` is a claim. The `scope` field makes it
impossible to quote the number without the scope.

Every number carries its evidence tag. `[V]` measured, `[U]` not measured, `[A]` derived or
assumed. A reviewer's most damaging finding is a projection presented as a measurement, and the
tag makes that a mechanical check rather than a judgement call.

## Schema

```yaml
- id: block.encrypted_evaluation     # stable, dotted, referenced from figures and lint
  value: 652.545
  unit: s
  tag: "[V]"                         # [V] measured | [U] unresolved | [A] derived/assumed
  scope: "Encrypted evaluation only; 98.4% of wall clock"
  source: "docs/hybrid/t123_walkthrough.md §8"
  notes: "optional"                  # caveats, alternative values, what not to conflate it with
```

`reference` and `reference_name` appear on baseline rows to record the published comparison.

## Rules

1. **A number not in the ledger does not go in the paper.** Add it here with a source first.
2. **Never change a `[V]` value without a new measurement.** Correcting a transcription error is
   fine; adjusting a number to fit prose is not.
3. **Never promote a tag.** `[A]` becomes `[V]` only when a measurement exists.
4. **No repository bookkeeping.** No SHA-256 hashes, run identifiers, manifest rows, fixture
   names, host names, or scheduler job numbers. Those prove results to the repository; they are
   not scientific content.
5. **Derived values name their inputs** in `source`, so the arithmetic can be rechecked.
6. **Open provenance is tracked, not hidden.** `optimizations.yaml` has an `open_provenance`
   block for numbers that are attested but not yet documented with a source. Empty it before
   submission, or restate the claim.

## Checking

```bash
python3 paper-docs/scripts/check_numbers.py
```

Fails if the manuscript contains a numeric value absent from the ledger, or a banned internal
term. Run it before every commit that touches `manuscript/`.
