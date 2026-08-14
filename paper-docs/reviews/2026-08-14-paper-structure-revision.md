# Paper-structure revision — 2026-08-14

## Scope

This pass implements the no-new-execution fallback from the academic review. It changes manuscript
argument, scope, tables, and float containment only. It does not modify evidence ledgers, figure
generation, bibliography records, repository documentation, or accepted run artifacts.

## Result

The manuscript now presents a bounded applied-HE feasibility claim:

- all twelve released blocks and the task head were executed at the 103-token prompt;
- graph feasibility is established for the complete released computation;
- numerical agreement is reported for one deterministically selected input, not generalized to
  encrypted task accuracy;
- practicality is negative for the current single-process interactive implementation;
- networked service performance, serialized bytes, throughput, and other-input stability remain
  unmeasured.

The abstract and Key Points were shortened, and the Introduction now gives three contributions:
complete-model feasibility, model-faithful reproducibility, and measured systems anatomy with
scoped negative results.

## Claim corrections

- Replaced whole-LayerNorm client wording with the implemented allocation: the server computes the
  statistics and affine work, and the client evaluates inverse square root only.
- Replaced uninterrupted encrypted-lineage wording with one key lineage plus declared client
  refreshes.
- Described the three-link reference chain: released float32 diagnostic lineage; NumPy float64
  versus a float64-cast PyTorch shadow; encrypted execution versus NumPy float64.
- Corrected complete-margin headroom to 6.67 orders of magnitude.
- Removed unmeasured sub-second plaintext CPU claims and affirmative batch/offline-suitability
  claims.
- Recast sampled GPU utilization as underutilization rather than causal bottleneck evidence.
- Qualified THOR and MOAI as uncontrolled cross-system context rather than a matched comparison.
- Narrowed model-extraction language to a chosen-query surface and possible identifiability of some
  affine segments.
- Framed the full 22,604-example genomic-signal evaluation as release/pipeline fidelity over a
  corpus that includes head-training examples, not held-out performance.

## Deployment and communication

The deployment table now includes ordinary plaintext service alongside client-local inference and
client-assisted CKKS. The scenario states that the evaluated path assumes distribution of the
tokenizer and embedding table; private lookup remains necessary when those components cannot be
distributed.

Results now contain one evaluation-setup subsection that records deterministic input selection,
single-input and single-run scope, hardware, timing endpoints, and the in-process role emulation.
A derived communication table reports ciphertext objects by direction, endpoints, and the minimum
sequential dependency phases. Serialized byte volume is explicitly unknown because no wire format
or network transport was measured.

## Literature positioning

The related-work structure now includes THE-X as an interactive predecessor, Powerformer as a
model-modifying HE-transformer direction, and ELLMo as packing/depth-aware optimization context.
The approximate-HE limitation cites Li--Micciancio when explaining why adversarial decryption
queries and noise flooding require a separate security analysis.

## Presentation

The heterogeneous plaintext baseline figure was removed from the manuscript; its qualified table
remains. A forced page flush after the protocol algorithms and at the end of the protocol section
contains Algorithms 1--4 before Section 6 and the bibliography. In the rebuilt 26-page PDF,
Algorithms 1--2 occur on page 10, Algorithms 3--4 on page 11, Section 6 begins on page 14, and the
references begin on page 24.

## Verification

`paper-docs/scripts/build.sh pdf` succeeds with Tectonic. The existing third-party-package UTF-8
warnings remain non-fatal.

`paper-docs/scripts/build.sh lint` reports clean banned-term, numeric-evidence, and citation checks.
Its only current structure finding is that `fig_baseline.pdf` remains generated although the
manuscript no longer includes it. Removing that figure from the generator/output belongs to the
concurrent figures revision and is intentionally not changed here.
