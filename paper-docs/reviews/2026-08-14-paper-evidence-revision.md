# Evidence revision audit — 2026-08-14

Scope: evidence and context changes needed for the manuscript revision without a new execution. This
is a verification report only. It does not edit the ledger, manuscript, figures, or accepted result
artifacts.

## Verdict

The existing artifacts support the complete-model feasibility claim, the corrected numerical chain,
the evaluated-input selection rule, and exact ciphertext-object counts by direction. They also support
an algorithmic lower bound on sequential client/server dependency phases. They do **not** support
serialized byte volume, network latency, throughput, or a deployed client/server claim.

## Exact `measurements.yaml` changes

Keep existing stable IDs where possible. Change scopes without changing measured values.

| Key | Required change | Tag and source |
|---|---|---|
| `oracle.reference_vs_upstream_tolerance` | Retain `2e-5`, but change the scope to “NumPy float64 reference versus a float64-cast shadow of the released PyTorch model, per block and at the head.” It is not a gate against the released float32 path. | `[V]`; `fhe/multiblock/export_fixture_t103.py:153-156,182-197,203-207,417-424` |
| `full.label_matches` | Change the scope to “encrypted label matches the manual NumPy float64 label.” Remove “Phase-A oracle” from this row. | `[V]`; complete-run JSON `head.label_matches` and driver `:1171-1178` |
| `full.margin_rel_error` | Retain `8.56e-9`; state explicitly that it compares the encrypted margin with the manual float64 margin. Replace “nine orders” with “6.67 orders” or “roughly seven orders.” | `[V]`; complete-run JSON `head.margin_rel_error` |
| `compose.twelve_block_at_task_length` | Replace “one encrypted lineage” with “one key lineage with eleven full-hidden-state client refreshes and one last-token refresh into the head.” | `[V]`; complete-run JSON `composition`; all-block driver `:1096-1129` |
| `full.crossings_nonlinearity` | Clarify that `10,286` is the shared-client counter: twelve blocks' nonlinear boundaries plus the two head inverse-square-root calls. It excludes eleven inter-block refreshes, the head-input refresh, and head SiLU. | `[V]`; complete-run JSON `protocol`; all-block driver `:121-140` |
| `base.gsr_accuracy` | Scope as evaluation on the full 22,604-example balanced source corpus, not a held-out or canonical test split. Treat it as release/pipeline-fidelity evidence. | `[V]`; `results/shared/manifest.yaml` dataset fields; `docs/tasks.md` setup; DNAGPT split audit in `reviews/2026-08-14-paper-sources.md` |
| `base.gsr_n` | Replace “Canonical test split size” with “Full balanced source-corpus size evaluated locally; not a held-out split.” | `[V]`; same sources as `base.gsr_accuracy` |
| `boundary.layernorm_instances` | Expand the scope: the server computes mean, centered square, variance, normalization multiplication, and scale; the client evaluates only inverse square root. | `[A]`; `context/08_implementation_ground_truth.md` Algorithm 2 and block driver `:514-538` |

Add these rows:

| New key | Value | Tag | Scope and source |
|---|---:|---|---|
| `oracle.numpy_vs_float64_shadow_max_block_abs` | `5.22e-6` | `[V]` | Largest per-block absolute difference in the frozen T=103 fixture; fixture manifest `oracle_gate.per_block[].max_abs_error_vs_float64_shadow_upstream`. |
| `oracle.numpy_vs_released_float32_max_block_abs` | `7.997e-4` | `[V]` | Largest diagnostic per-block absolute drift, reached after block 11; fixture manifest `oracle_gate.per_block[].diagnostic_max_abs_error_vs_float32_upstream`. |
| `oracle.released_float32_margin` | `12.2666015625` | `[V]` | Released float32 full-prompt margin; fixture manifest `classification.complete_public_prompt_reference.margin_n_minus_a`. |
| `oracle.released_float32_label` | `N` | `[V]` | Released float32 full-prompt label; same manifest object. |
| `full.encrypted_margin` | `12.266581857561365` | `[V]` | Decrypted encrypted-execution margin; complete-run JSON `head.decrypted_margin_n_minus_a`. |
| `full.encrypted_label` | `N` | `[V]` | Complete-run JSON `head.label`. |
| `full.margin_rel_error_vs_released_float32` | `1.61e-6` | `[A]` | `abs(full.encrypted_margin - oracle.released_float32_margin) / abs(oracle.released_float32_margin)`; exact derived value `1.6063893928e-6`. |
| `full.margin_headroom_ratio` | `4.67e6` | `[A]` | `4e-2 / 8.55814015132133e-9`. |
| `full.margin_headroom_orders` | `6.67` | `[A]` | `log10(full.margin_headroom_ratio)`. Do not alter `err.headroom_orders`, which concerns the separate one-block hidden-state error. |
| `full.encrypted_examples` | `1` | `[V]` | One selected prompt was evaluated in the complete run; complete-run artifact plus fixture manifest. |
| `input.gsr_record_index` | `0` | `[V]` | Exporter deterministically selects `records[0]` from the verified positive GSR FASTA; `export_fixture_t103.py:139-141,374-382`. |
| `input.gsr_selection_rule` | `first record in the verified positive GSR FASTA` | `[V]` | Same source. State that selection was deterministic, not random, stratified, or margin-based. |
| `input.gsr_class` | `positive / real GSR` | `[V]` | Fixture manifest `classification.*.binary_meaning`. |
| `boundary.layernorm_client_function` | `inverse square root only` | `[V]` | Block driver `:514-538`; implementation ground truth Algorithm 2. |

The three-link reference chain should then be described as:

1. released float32 inference versus the independent/manual lineage, with recorded drift and label
   agreement;
2. independent NumPy float64 versus the float64-cast PyTorch shadow at `2e-5`;
3. encrypted execution versus the NumPy float64 reference.

The float32 path, float64 manual path, and encrypted path all label this prompt `N`. Only the second
and third links are tolerance gates; float32 drift is a disclosed diagnostic.

## Communication accounting supportable without a run

One block sends 130 ciphertext objects server-to-client: 26 LayerNorm inverse-square-root inputs, 91
attention score tiles, and 13 GELU inputs. It sends 766 client-to-server: 26 LayerNorm inverses, 727
attention-weight tiles, and 13 GELU outputs. The block source establishes the one-ciphertext payload of
each boundary method (`:514-715`), while the ledger already records the `26/91/727/13` schedule.

Across twelve blocks, eleven 13-ciphertext full-state refreshes, one 13-to-1 head-input refresh, and
three one-to-one head nonlinearities, add these `[A]` rows:

| New key | Value | Scope |
|---|---:|---|
| `comm.block_server_to_client_ciphertexts` | `130` | One transformer block. |
| `comm.block_client_to_server_ciphertexts` | `766` | One transformer block. |
| `comm.full_server_to_client_boundary_ciphertexts` | `1719` | All nonlinear and refresh boundaries; excludes final output. |
| `comm.full_client_to_server_boundary_ciphertexts` | `9339` | All nonlinear and refresh boundaries; excludes initial input. |
| `comm.full_boundary_ciphertexts_total` | `11058` | Sum of the preceding two rows. |
| `comm.initial_input_ciphertexts` | `13` | Client-to-server embedded-input groups; driver input encryption and `layout.token_groups`. |
| `comm.final_output_ciphertexts` | `1` | Server-to-client encrypted classification margin. |
| `comm.full_ciphertexts_including_endpoints` | `11072` | `11058 + 13 + 1`. |
| `comm.minimum_sequential_boundary_phases` | `63` | Ideal same-stage batching under the evaluated protocol: four per block, eleven inter-block refreshes, one head-input refresh, and three head nonlinearities. This is an algorithmic lower bound, not an observed RPC count. |
| `comm.minimum_transport_phases_including_endpoints` | `65` | The preceding 63 plus initial input upload and final result return. |
| `comm.networked_implementation` | `false` | `[V]`; current roles share one process and node. |
| `comm.serialized_bytes_measured` | `false` | `[V]`; no wire serialization occurs in the complete run. |

The 63-phase derivation assumes independent calls at one protocol stage can be batched into one
request/response: LayerNorm 1, attention scores/weights, LayerNorm 2, and GELU form four sequential
phases per block. It does not claim the present implementation performs that batching.

### Bytes cannot be derived honestly

Do not add a byte-volume number. The complete artifact records neither serialized boundary objects
nor their sizes by level. FIDESlib's evaluated API lacked ciphertext serialization in the documented
cross-process attempt (`docs/hybrid/tasks.md:2652-2671`). A coefficient-array estimate would require
unrecorded boundary levels, component counts, host representation, compression, and serialization
overhead; it would not be wire bytes. If a ledger placeholder is desired, use
`comm.serialized_byte_volume: unknown`, tag `[U]`, with those reasons.

## Full-run and status documents

These live sources contradict the accepted 2026-08-12 complete run and must be updated in the same
revision:

- `AGENTS.md` current state and next-step paragraph;
- `CLAUDE.md` GPU-execution sentence saying the driver may now run for arithmetic closure;
- `docs/roadmap.md` verified foundation and Stages 3–4;
- `docs/shared/feasibility_overview.md` strongest claims, unresolved claims, and execution decision;
- `docs/tasks.md` Phase-B closing paragraph, plus “full test set” in the GSR verdict;
- `paper-docs/AGENTS.md` Rule 2's one-block measured-unit/projection language;
- `paper-docs/context/00_terminology.md` task-length-block status;
- `paper-docs/context/02_research_timeline.md`, `03_methods_and_protocol.md`,
  `04_results_and_limits.md`, `05_claims_and_qualifiers.md`, and `06_defensibility.md` wherever they
  say all-block execution, a final label, or dedicated timing remain unmeasured.

For historical detailed records, preserve the old result text and add a dated supersession note
instead of rewriting history: `docs/hybrid/tasks.md` entries before 2026-08-12,
`docs/hybrid/t123_walkthrough.md`, `docs/hybrid/optimizations_and_combinations_report.md`, and
`paper-docs/context/README_sourcebook.md`. The last file already identifies itself as historical.

The current unresolved boundaries are: no independent complete-run repeat, one evaluated prompt, no
encrypted task-set accuracy, no serialized or networked transport, no private token lookup, and no
production security evaluation.
