# Final paper-evidence verification — 2026-08-14

## Outcome

The no-rerun revision is materially stronger, and paper lint is clean. One blocker and two high-priority evidence-scope issues remain before the manuscript is evidence-clean. Three medium-priority wording/measurement-scope corrections should also be made. None requires rerunning the experiment.

## Blocker

### B1. The `client_instances_logical` counter is heterogeneous, not a scalar count of nonlinear values

- The ledger defines `ops.client_instances_logical: 129162` as individual nonlinear values and derives the 99.5% softmax share from it (`paper-docs/evidence/measurements.yaml:276-283`, `paper-docs/evidence/measurements.yaml:297-327`). The full-run entry similarly describes `1551081` as individual nonlinear values (`paper-docs/evidence/measurements.yaml:970-974`).
- The manuscript repeats those interpretations (`paper-docs/sections/05_protocol.tex:194-200`, `paper-docs/sections/07_results.tex:71-85`, `paper-docs/sections/07_results.tex:235-238`).
- The implementation increments this counter at different granularities: LayerNorm by active tokens and GELU by active token-copies (`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp:514-565`), while attention scores and attention weights are counted per head/query/key scalar (`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp:583-625`, `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp:667-713`). The full-run total additionally includes refresh-token instances.

Required correction: rename the ledger quantity as an implementation-declared heterogeneous logical-instance counter and remove the claim that it counts individual nonlinear values. Do not use its 99.5% split as a scalar-work, traffic, or cost share. If a dominance statistic is wanted, use the homogeneous communication accounting already present: attention contributes 818 of 857 block client calls (95.4%) or 818 of 896 block ciphertext transmissions (91.3%). Describe the full-run value with the same heterogeneous scope and state that refresh instances are included.

## High priority

### H1. “Uncontended node” is stronger than the retained telemetry

- The telemetry verifies a whole-node allocation and an empty node at allocation, but the job did not request scheduler exclusivity; absence of other GPU or CPU activity is partly inferred, and total allocated CPU capacity is not retained (`paper-docs/evidence/telemetry.yaml:102-140`).
- The categorical wording appears in the results selection paragraph, full-run table caption, limitations, and scaling ledger (`paper-docs/sections/07_results.tex:14-25`, `paper-docs/sections/07_results.tex:201-205`, `paper-docs/sections/08_limitations.tex:10-13`, `paper-docs/evidence/scaling.yaml:109-111`).

Required correction: replace “uncontended node” with “whole-node allocation with no contention detected in the recorded telemetry,” or equivalent. This preserves the useful run-quality qualification without turning an observed absence into a scheduler guarantee.

### H2. The new float64/float32 reference claims depend on a gitignored fixture manifest

- The ledger’s float64-versus-float32 deviation, float64 plaintext-oracle error, released float32 margin, and derived end-to-end error chain cite the T=103 reference manifest (`paper-docs/evidence/measurements.yaml:365-416`).
- That manifest is under `checkpoints/fhe_exports/`, which is excluded by the repository policy (`.gitignore:89`), so a clean checkout cannot verify the cited `5.22e-6` and `7.997e-4` values or the accompanying label metadata.
- The released float32 margin is independently retained in `results/runs/fhe_multiblock_plaintext_contract_20260724.json:39`, and the deterministic record-0 selection is supported by `eval/export_fixture_t103.py:139-141` and `eval/export_fixture_t103.py:374-382`.

Required correction: repoint the released float32 margin to the committed run record. Add a small committed, metadata-only accepted evidence record for the float32/float64 comparison and label checks, excluding weights, sequence/token data, and prohibited fixture material, then cite that record from the relevant YAML keys. This is evidence preservation, not a rerun.

## Medium priority

### M1. The 9,839 MiB sample is device-wide GPU memory, not a direct per-process metric

- The retained sampler uses device `memory.used`; nevertheless, the ledger calls it the evaluating process’s peak (`paper-docs/evidence/measurements.yaml:1012-1019`) and the results use process-footprint wording (`paper-docs/sections/07_results.tex:121-124`, `paper-docs/sections/07_results.tex:148-154`). Empty-node evidence makes attribution reasonable but does not change the metric’s scope.
- The ledger also says the value is flat across the run, whereas telemetry says it is reached at about 400 seconds and remains flat for the remaining 94% (`paper-docs/evidence/measurements.yaml:1016-1018`, `paper-docs/evidence/telemetry.yaml:149-156`).

Required correction: call it “evaluating-device memory used during the run,” state the attribution to the run separately, and reuse the telemetry’s precise plateau wording.

### M2. `server_linear_algebra_seconds` is a residual provider-side bucket, not pure GPU linear algebra

- The implementation computes it as encrypted evaluation time minus client time (`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3.cpp:953-954`). It therefore includes all non-client work inside the encrypted evaluation interval, including host preparation, synchronization, and evaluator overhead.
- The narrower interpretation appears in the ledger, manuscript, and cost figure source (`paper-docs/evidence/measurements.yaml:796-807`, `paper-docs/sections/07_results.tex:192-197`, `paper-docs/sections/07_results.tex:214-216`, `paper-docs/sections/07_results.tex:232-235`, `paper-docs/scripts/figures.py:864-979`).

Required correction: rename the bucket “provider-side encrypted evaluation (GPU backend plus host support)” or equivalent. Keep the exact 5,638.561-second measurement, but do not present it as GPU-kernel-only time.

### M3. Keep the communication result object-count-only

The derived accounting is internally consistent: 1,719 server-to-client plus 9,339 client-to-server ciphertext objects gives 11,058 boundary objects; adding 13 encrypted inputs and one output gives 11,072 total encrypted objects. The 63 ideal boundary phases, or 65 including endpoints, are also arithmetically supported (`paper-docs/evidence/measurements.yaml:1041-1124`, `paper-docs/sections/07_results.tex:93-119`). Serialized byte volume remains unresolved. Preserve that distinction and avoid converting object counts into bandwidth or network-time claims without serialization or network measurements.

## Verified clean

- The corrected `6.67`-order headroom and `4.67e6` ratio agree with the cited margin and maximum error (`paper-docs/evidence/measurements.yaml:749-768`, `paper-docs/sections/00_frontmatter.tex:46-48`, `paper-docs/sections/07_results.tex:42-55`).
- The manuscript now distinguishes the released float32 prediction, float64 exported oracle, and encrypted output, and it does not compare the encrypted result directly to the released float32 margin (`paper-docs/sections/04_plaintext_baseline.tex:24-34`, `paper-docs/sections/07_results.tex:29-47`).
- The single record-0 arithmetic input is disclosed, and no distributional generalization is claimed (`paper-docs/evidence/measurements.yaml:417-433`, `paper-docs/sections/07_results.tex:14-25`, `paper-docs/sections/08_limitations.tex:10-17`).
- LayerNorm placement is consistent across protocol prose, ledger, and architecture figure: server-side mean/variance and affine work; client-side inverse square root (`paper-docs/evidence/measurements.yaml:300-307`, `paper-docs/evidence/measurements.yaml:434-438`, `paper-docs/sections/05_protocol.tex:34-47`, `paper-docs/sections/05_protocol.tex:69-75`, `paper-docs/scripts/figures.py:252-265`).
- GSR wording distinguishes the full 2,944-locus released-model task from the single-input encrypted arithmetic run (`paper-docs/sections/04_plaintext_baseline.tex:24-34`, `paper-docs/sections/07_results.tex:14-25`).
- The full-run latency, memory-margin arithmetic, ciphertext-object counts, and figure captions are numerically consistent, subject to the scope corrections above. The inactive baseline/waterfall generators are not registered for manuscript figure generation (`paper-docs/scripts/figures.py:1197-1209`).
- `paper-docs/scripts/build.sh lint` passes banned-term, number, citation, and structure checks.

## Closure check after style and evidence edits

### Verdict

- **Active manuscript and evidence ledgers: PASS.** All six findings above are resolved, and the style pass did not change the scope or arithmetic of the load-bearing claims.
- **Repository documentation closure: FAIL.** Several canonical context pages still describe the pre-2026-08-12 state and would reintroduce false claims if used as the paper sourcebook. The new precision record is correct and registered, but it is still untracked in the reviewed worktree.

### Resolution of the six findings

1. **Heterogeneous counter — resolved.** The ledger now defines both block and full-run values as mixed-unit schedule counters and expressly disclaims scalar-work, traffic, and cost interpretations (`paper-docs/evidence/measurements.yaml:281-356`, `paper-docs/evidence/measurements.yaml:1006-1013`). The manuscript uses the homogeneous 818/857 call share and 818/896 object share instead (`paper-docs/manuscript/source/sections/05_protocol.tex:194-200`, `paper-docs/manuscript/source/sections/07_results.tex:73-89`).
2. **Whole-node scope — resolved.** The paper says that no contention was detected in retained telemetry and discloses that scheduler exclusivity was not requested (`paper-docs/manuscript/source/sections/07_results.tex:21-27`, `paper-docs/manuscript/source/sections/07_results.tex:205-210`, `paper-docs/manuscript/source/sections/08_limitations.tex:10-13`). The ledgers use the same qualification (`paper-docs/evidence/measurements.yaml:15-23`, `paper-docs/evidence/measurements.yaml:740-743`, `paper-docs/evidence/scaling.yaml:108-112`, `paper-docs/evidence/telemetry.yaml:82-85`).
3. **Precision metadata and provenance — resolved in content.** The metadata-only record preserves the two precision deviations, both reference margins and labels, the deterministic record-0 selection, exporter, source-manifest SHA-256, and exclusion of weights/input arrays (`results/runs/fhe_multiblock_t103_precision_contract_20260814.json:1-34`). It is registered in the shared manifest (`results/shared/manifest.yaml:98-120`) and cited by the measurement ledger (`paper-docs/evidence/measurements.yaml:404-449`). Recomputed SHA-256 values match both manifest provenance fields exactly: `12d96c...3996` for the preserved JSON and `ab5553...962d` for the ignored source manifest.
4. **GPU-memory scope — resolved in the active paper and ledger.** The measurement is called device-wide `memory.used`, its attribution is qualified, and the plateau is scoped to after roughly 400 seconds/final 94% (`paper-docs/evidence/measurements.yaml:1052-1060`, `paper-docs/manuscript/source/sections/07_results.tex:124-128`, `paper-docs/manuscript/source/sections/07_results.tex:142-155`).
5. **Provider residual timing — resolved.** The ledger defines the 5,638.561-second bucket as the residual provider-side encrypted-evaluation interval, including GPU-backend work and host support (`paper-docs/evidence/measurements.yaml:830-837`). The cost caption and prose repeat that scope and state that it is not GPU-kernel time (`paper-docs/manuscript/source/sections/07_results.tex:193-201`, `paper-docs/manuscript/source/sections/07_results.tex:219-242`); the generated figure labels it “GPU backend + host support” (`paper-docs/scripts/figures.py:959-964`).
6. **Communication scope — resolved.** The paper presents derived ciphertext-object counts and algorithmic lower-bound phases, not measured messages or bytes, and leaves serialized volume unknown (`paper-docs/evidence/measurements.yaml:1105-1166`, `paper-docs/manuscript/source/sections/07_results.tex:96-122`, `paper-docs/manuscript/source/sections/08_limitations.tex:30-34`).

### Style-edit integrity

The headline values remain consistent with the ledgers: one selected 103-token prompt; all 12 blocks plus the released head; margin relative error `8.56e-9`; tolerance `4e-2`; exact headroom `6.67` orders; wall time `6683 s`/`1.86 h`; encrypted evaluation `6594.748 s`; provider residual `5638.561 s`; client work `956.187 s`; device-wide GPU peak `9839 MiB`; host peak `50930 MiB`; and object counts `1719 + 9339 = 11058`, or `11072` including endpoints. The abstract, baseline, results, limitations, conclusion, and active captions preserve the single-input/single-execution and non-networked qualifications. `paper-docs/scripts/build.sh lint` again passes banned terms, evidence-number checks, citations, and structure.

### Exact remaining items

1. **Stale canonical status pages.** These still say that the complete 103-token run is pending or unexecuted: `paper-docs/context/02_research_timeline.md:209-221`, `paper-docs/context/03_methods_and_protocol.md:184-192`, `paper-docs/context/04_results_and_limits.md:199-215`, `paper-docs/context/05_claims_and_qualifiers.md:98-103`, `paper-docs/context/05_claims_and_qualifiers.md:156-167`, `paper-docs/context/06_defensibility.md:68-90`, `paper-docs/context/06_defensibility.md:355-386`, and `paper-docs/context/06_defensibility.md:521-523`. They should be updated to the same single-input, single-execution complete-model result now used by the manuscript. This is the only substantive closure failure.
2. **Stale device-memory terminology in the sourcebook.** One-block summaries still call a device-wide sample a process footprint (`paper-docs/context/05_claims_and_qualifiers.md:79`, `paper-docs/context/05_claims_and_qualifiers.md:103`, `paper-docs/context/06_defensibility.md:76-82`, `paper-docs/context/06_defensibility.md:142-144`). Change only the metric label and retain the relevant one-block value/scope.
3. **Worktree integration.** `results/runs/fhe_multiblock_t103_precision_contract_20260814.json` is not ignored and its manifest hash is correct, but `git ls-files` shows it is not yet tracked. Ensure that file is included with the eventual repository change; otherwise the precision-chain closure will disappear from a clean checkout.

## Final repository-document closure recheck

**FAIL by one remaining line; this paragraph supersedes the preceding “Exact remaining items” list.** Context pages 02–06 no longer contain a pending/unexecuted complete-run or final-label contradiction, and their one-input, one-execution, no-task-accuracy limits are consistent. The sole remaining scope error is `paper-docs/context/06_defensibility.md:214-215`, which says the GPU memory telemetry is “process-specific rather than device-wide”; the retained GPU samples and the same sourcebook's result tables are device-wide, while host RSS is process-specific. The precision JSON parses successfully, its SHA-256 (`12d96c...3996`) matches `results/shared/manifest.yaml:113`, and the ignored source-manifest SHA-256 (`ab5553...962d`) matches `results/shared/manifest.yaml:114`; its current untracked state is therefore only a handoff/integration concern, not an evidence-content defect.
