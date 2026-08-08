# Source map: where each paper claim comes from

The repository is large and most of it is not paper material. This table says which file answers
which manuscript question, so an agent does not have to search, and does not have to guess.

**Rule:** numbers come from `paper-docs/evidence/*.yaml`. Those files cite the repository sources
below. If a number is not in the ledger, it does not go in the paper — add it to the ledger with
its source first.

## By manuscript section

| Section | Primary source | Secondary |
|---|---|---|
| §1 Introduction | `context/01_scenario_and_motivation.md` | `context/02_research_timeline.md` |
| §2 Background | `docs/shared/backend_selection.md` | `docs/overview.md` |
| §3 Scenario and threat model | `context/01_scenario_and_motivation.md`, `context/06_defensibility.md` §3.1 | `context/03_methods_and_protocol.md` |
| §4 Plaintext baseline | `docs/tasks.md`, `docs/eval_approach.md` | `docs/data_provenance.md` |
| §5 Protocol | **`context/08_implementation_ground_truth.md`** (pseudocode, layout, mechanisms) | `context/03_methods_and_protocol.md`, `docs/shared/architecture_options.md` |
| §6 Optimization | `evidence/optimizations.yaml`, **`context/08_implementation_ground_truth.md`** §5 | `docs/hybrid/optimizations_and_combinations_report.md` |
| §7 Results | `evidence/measurements.yaml`, `evidence/telemetry.yaml`, `evidence/scaling.yaml` | `docs/hybrid/t123_walkthrough.md` §8 |
| §8 Limitations | `context/05_claims_and_qualifiers.md`, `docs/hybrid/t123_walkthrough.md` §11 | `docs/hybrid/roadmap.md` |
| §9 Related work | `docs/shared/backend_selection.md`, `docs/hybrid/roadmap.md` (THOR, HE-BLAS, AEGIS refs) | — |
| §10 Conclusion / future work | `docs/hybrid/roadmap.md`, `evidence/optimizations.yaml` `open:` | `context/06_defensibility.md` NEXT STEPS |

## By question

| Question | Owning file |
|---|---|
| What is the project trying to establish? | `docs/overview.md` |
| Why validate the model in plaintext first? | `docs/eval_approach.md` |
| What are the plaintext task results, and against what references? | `docs/tasks.md` |
| Where did the datasets come from, and under what licence? | `docs/data_provenance.md` |
| Why CKKS, and why this GPU backend? | `docs/shared/backend_selection.md` |
| Why is the architecture client-assisted? | `docs/shared/architecture_options.md` |
| What did the non-interactive configuration establish, and where did it stop? | `docs/pure/measurements.md` |
| How does one encrypted block actually work, end to end? | `context/08_implementation_ground_truth.md` (verified against source), then `docs/hybrid/t123_walkthrough.md` |
| What is the pseudocode for the protocol? | `context/08_implementation_ground_truth.md` §4 |
| Do the published operation counts actually derive? | `context/08_implementation_ground_truth.md` §3 — all twelve re-derived and confirmed |
| What validates the plaintext reference itself? | `context/08_implementation_ground_truth.md` §1 — the two-stage chain |
| What is the packing layout? | `docs/hybrid/t123_walkthrough.md` §4 |
| Why is there any ciphertext-ciphertext multiplication if weights are public? | `docs/hybrid/t123_walkthrough.md` §3.1 |
| Which optimizations worked, failed, or remain open? | `docs/hybrid/optimizations_and_combinations_report.md`, `evidence/optimizations.yaml` |
| What is measured versus projected? | `evidence/*.yaml` — read the `tag` field |
| What must not be claimed? | `context/05_claims_and_qualifiers.md`, `context/00_terminology.md` |
| What runs next? | `docs/hybrid/roadmap.md` |

## Files that are *not* paper sources

Do not cite, quote, or draw numbers from these. They are development infrastructure and their
contents do not belong in a scientific narrative.

- `results/**` — run JSONs and manifests. Superseded for paper purposes by `evidence/*.yaml`.
- `fhe/**` — implementation sources, launchers, contract tests.
- `kimon/**` — environment build scripts, scheduler configs, platform contracts.
- `docker/**` — pinned environment definitions.
- Any file whose content is hashes, fixture names, or job identifiers.

The one exception: a mechanism the manuscript describes must be verified against the
implementation rather than against prose. That verification has been done and written up in
[`08_implementation_ground_truth.md`](08_implementation_ground_truth.md), with `file:line`
citations, the protocol pseudocode, an independent re-derivation of every operation count, and a
claims audit. Use that note instead of re-reading the sources, and cite the mechanism rather than
the filename.

## Reference works to cite

Grounded in the repository docs, and each one needs verification against the live record before
it enters `refs.bib`.

| Work | Where the repo relies on it |
|---|---|
| DNAGPT (arXiv 2307.05628) | the model under study |
| DNABERT-2 | GUE promoter and splice-site references |
| DeepGSR | genomic-signal recognition reference |
| Xpresso | mRNA abundance split and reference |
| GUE benchmark | task family definition |
| CKKS | the encryption scheme |
| OpenFHE | correctness semantics |
| FIDESlib | the evaluated GPU backend |
| THOR | compact packing and specialized HE matrix algorithms |
| Homomorphic linear algebra with BLAS (arXiv 2503.16080) | matrix-primitive design evidence |
| AEGIS (arXiv 2604.03425) | multi-GPU HE placement and overlap; design evidence only |
| Concrete-ML / TFHE-rs | evaluated and rejected backend |

Prior private-transformer-inference work (Iron, BOLT, BumbleBee, Nimbus, and the CKKS-transformer
line) is **not** yet grounded in the repository. It has to be located and read before §9 can be
written; do not let a citation into the bibliography on the strength of a remembered title.
