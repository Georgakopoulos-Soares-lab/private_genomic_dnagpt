# Evidence audit — 2026-08-13

Scope: full audit of `manuscript/source/` after the 2026-08-12 complete-model run was banked.
Every complete-model value was re-derived from the primary artifacts, not from the ledger:

- `results/runs/fhe_fides_real_d768_t103_12blocks_head_cpudiagcache_t123_v3_scheme_b_a100_20260812.json`
- `..._telemetry.csv` (653 samples), `..._slurm.out`, `..._slurm.err`, `..._run.log`
- `kimon/configs/config3_all_opts/e2e_t123_v3_monitored.sbatch`

`python3 paper-docs/scripts/check_numbers.py` → all four checks clean. The lint is permissive by
construction (it accepts any digit string appearing anywhere in a ledger `scope`/`notes`/`statement`
field, plus every 0–5 decimal-place rounding of every float), so a clean lint carries little weight
on scope. The findings below come from reading.

## Findings, most severe first

| # | Location | Claim as written | Problem | Fix |
|---|---|---|---|---|
| 1 | `07_results.tex:238` | "That is roughly seven orders of magnitude above the plaintext forward pass for the same input" | **Unbacked number, and almost certainly wrong.** No ledger row exists for a plaintext forward-pass time or a slowdown ratio (`grep` of `evidence/*.yaml` returns only `err.headroom_orders`, which is the *correctness headroom*, an unrelated quantity that happens to also be "seven"). Seven orders would put the plaintext pass at 0.67 ms; 12 blocks x 768 width x 103 tokens is ~17.5 GFLOP, i.e. ms on an A100 and ~10^2 ms on CPU — 5 to 6 orders, not 7. This is the one number a reviewer can falsify on their own laptop. | Delete the sentence, or measure the plaintext forward pass on the same host, bank it as a `[V]` row, and quote the derived ratio as `[A]`. |
| 2 | `07_results.tex:114-119` (Table 3) | Column "Relative error": `One block, 103 tokens … 4.64e-9` / `Twelve blocks and task head, 103 tokens … 8.56e-9` | **Two different quantities in one column.** The block rows are hidden-state relative infinity norms over 103x768 values; `8.56e-9` is the *scalar classification-margin* relative error. Read down the column, the table asserts error did not grow across twelve blocks. The run JSON measures the comparable quantity and it did grow: `blocks[].refresh_metrics.global_rel_inf` runs 4.09e-9 (block 0) → 8.32e-9 (block 10) → 2.07e-8 (block 11, last-token refresh), and `max_abs_error` runs 6.56e-8 → 3.29e-6 → 7.97e-6, a ~100x growth in absolute deviation. None of this is in the ledger. | Split the column into "hidden-state rel. inf. norm" and "margin rel. error", bank the per-block `global_rel_inf`/`max_abs_error` series as `[V]`, and state the twelve-block hidden-state error explicitly. The claim survives — 2.07e-8 is still six orders inside 4e-2 — but it must be the stated number. |
| 3 | `00_frontmatter.tex:50,58`; `01_introduction.tex:71`; `07_results.tex:237`; `10_conclusion.tex:19` | "The complete inference **takes** $1.86$~hours"; "one encrypted classification **takes** / **occupies** an A100 GPU node for $1.86$~hours" | **n = 1 written in the present-habitual.** "takes"/"occupies" reads as a rate or a typical cost. §7.8 and §8 both say plainly that this is one execution, but the abstract, key points, introduction and conclusion — the four places a reviewer actually reads the number — do not. The ledger is explicit: `complete_model` header says "NOT REPRODUCED … Report it as one measured sample, never as a mean or a rate." | Past tense and one qualifier at first use in the abstract: "a single measured execution took 1.86 hours". Same in the key points and conclusion. |
| 4 | `00_frontmatter.tex:53-57` | "**Complete encrypted DNAGPT inference is therefore feasible.**" | **Scope overreach in the abstract.** `context/00_terminology.md` bans "end-to-end encrypted DNAGPT inference" because token-index lookup is outside the encrypted boundary; "Complete encrypted DNAGPT inference" carries the same reading, and the abstract contains no clause anywhere placing the encrypted scope at embedded vectors. §8 states it, but the abstract is what propagates. | One clause: "Complete encrypted inference over the released blocks and task head — beginning at embedded token vectors — is therefore feasible." |
| 5 | `00_frontmatter.tex:87`; `01_introduction.tex:72`; `10_conclusion.tex:23` | "GPU utilization stays **below** $51\%$" | **Contradicted by the data and internally inconsistent.** `max(gpu0_util_pct)` over 653 samples is exactly 51. §7.8:250 and the Fig. 4 caption both say "never exceeds 51%", which is correct. Three locations say "below", two say "never exceeds". | Use "never exceeds $51\%$" in all five places. |
| 6 | `07_results.tex:222`; `07_results.tex:96` (Fig. 4 caption) | "GPU memory stays flat at $9839$~MiB **for the entire run**"; "GPU memory is flat **throughout**" | **False for the first ~400 s.** `gpu0_mem_used_mib` steps 0 → 425 (t=42) → 7791 (t=52) → 8815 (t=103) → 9839 (t=400), then is constant to t=6675. The correct claim — peak reached inside block 0 and no growth with block index — is the one that matters and is fully supported. `telemetry.yaml:123-129` (`clean.memory_flat_across_blocks`) carries the same overstatement and should be corrected too. | "GPU memory reaches $9839$~MiB during the first block and does not grow with block index." §7.3:100-101 already words this correctly; copy it. |
| 7 | `00_frontmatter.tex:51,79`; `01_introduction.tex:68`; `07_results.tex:79`; `10_conclusion.tex:14` vs `07_results.tex:101,222` | "$9.6$~GiB" (5 places) vs "$9839$~MiB" (2 places) | **Same quantity, two renderings.** 9839 MiB = 9.608 GiB. Additionally the two GiB uses in the abstract/intro/conclusion cite `mem.gpu_peak`, which is sourced to the *single-block* walkthrough, while the sentence they sit in is about the twelve-block run; the complete-model row is `full.gpu_peak_mib`. `9839` is also typeset without the thousands separator used by its neighbours (`50{,}930`, `48{,}610`). | Pick one rendering — `9839`~MiB, since the memory claim is now a complete-model claim — and apply it in all seven places, with the `{,}` separator. |
| 8 | `07_results.tex:149` (Table 4 caption) | "on one A100 GPU node with 32 CPU cores" | **Not a verified node property.** The batch script requests `-p gpu-a100 -N 1 -n 1 -c 32` with no `--exclusive`; `OMP_NUM_THREADS=32` is a thread count. `slurm.out` and `run.log` show the job enumerating **three** A100-PCIE-40GB devices. No committed artifact records the node's total core count, so "a node with 32 CPU cores" describes the request, not the hardware held. A reader costing this at node-hours is misled in the generous direction. | "on one node, using one NVIDIA A100 (40 GB) and 32 CPU threads." Or bank the node's core count as a `[V]` row from a recorded `lscpu`/`SLURM_CPUS_ON_NODE`. |
| 9 | `fig_cost_split.pdf` (`figures.py:869-871`) vs `07_results.tex:179-180` and `00_frontmatter.tex:84` | Figure bar labels `84.4%` / `14.3%`; prose "the compute provider therefore carries $85.5\%$ … the data owner $14.5\%$"; key points "$14.5\%$ of the encrypted work" | **Same split, two denominators, denominator unstated in the figure.** Figure percentages are of wall clock; prose percentages are of encrypted evaluation. Both are correct and both are in the ledger, but a reader comparing Fig. 3 with the sentence two paragraphs later sees 14.3 and 14.5 for the same bar. The figure's subtitle says "Server and client shares sum exactly to encrypted evaluation" while labelling them as shares of *wall*. | Label the figure segments with both, or state "% of wall clock" on the figure and keep the prose on encrypted evaluation with its denominator named in the sentence. |
| 10 | `fig_graphical_abstract.pdf` (`figures.py:221`) vs all prose | Graphical abstract prints `$8.6\times10^{-9}$`; abstract, key points, introduction, §7.1, §7.7 and conclusion all print `$8.56\times10^{-9}$` | Rounding mismatch between a figure and the text, in the single most-read element of the paper. The ledger's `full.margin_rel_error` note says "quote at three significant figures". | Use `8.56\times10^{-9}` in the graphical abstract. |
| 11 | `fig_timeline.pdf` (`figures.py:697-707`) | Dashed guide line and annotation "peak 51% — never approaches saturation" | The plotted series is `complete_model_trace`, thinned to 73 points, whose maximum `gpu_pct` is **41**. The annotated 51% guide floats 10 points above every plotted value, so the figure's own data contradicts its own annotation. | Either plot the full 653-sample series, or annotate the plotted maximum and move "51%" to the caption as a full-series statistic. |
| 12 | `06_optimization.tex:22-24` | "The repository does not contain the clean timing artifacts needed for an elapsed-time comparison" | **Now false as written**, and it contradicts §7, which reports 6683 s from a clean artifact. What is missing is a clean *pre-optimization baseline* (`open_provenance: prov.baseline_wall_time`), not clean timing artifacts in general. | "No dedicated-node measurement of the pre-optimization configuration exists, so this section reports the verified changes in work and memory." |
| 13 | `07_results.tex:188-190` | "Dividing block evaluation by twelve gives $547.166$~s per block, but this is an average within one run … so it carries **no information about variance**." | **Contradicted by the artifact.** The run JSON records twelve individual `evaluation_seconds`, from 533.869 s to 558.943 s (spread 25.07 s, 4.6%). These are twelve measurements of architecturally identical work from the *clean* run, so they are usable and they do bound within-run variation. The sentence claims less than the evidence supports and invites a reviewer who opens the JSON to conclude the authors did not look. | Bank the twelve per-block times as `[V]` and state the observed range, then keep the (correct) point that within-run spread is not run-to-run variance. |
| 14 | `06_optimization.tex:141` | "Mask plaintexts are still encoded afresh at $17{,}400$ uses per block." | **Tag downgrade.** The ledger row `open.mask_plaintext_encoding` is `[U]` — not measured — and says "approximately 17,400". The manuscript states it flat, with false precision, as if counted. | "approximately $17{,}400$ uses per block", or count it and re-tag. |
| 15 | `06_optimization.tex:149-151` | "the projected schedule reduces dense products from $156$ to $52$ … from $159{,}744$ to $53{,}248$; the total ciphertext–plaintext count falls by $60\%$" | "projected" is present (good), but the ledger row `open.distinct_transforms_per_copy` `[A]` hedges every one of these — "approximately 52", "approximately 53,248", "roughly 60 percent". The hedges were dropped, turning an unimplemented analysis into exact figures. | Restore "approximately"/"roughly". |
| 16 | `08_limitations.tex:44-46` | "The measured $4.64\times10^{-9}$ error leaves seven orders of magnitude inside the $4\times10^{-2}$ acceptance tolerance" | **Missing scope word.** This is the *single-block* error (`err.global_relative`, sourced to the single-block walkthrough). In §8, in a paragraph about the protocol's re-encryption path as a whole, it reads as the complete-model error. The complete-model figures are 8.56e-9 (margin) and 2.07e-8 (hidden state). | "the measured single-block error of $4.64\times10^{-9}$", or quote the complete-model number. |
| 17 | `07_results.tex:27-31` | "The decrypted output has relative infinity-norm error $4.64\times10^{-9}$ …" | Same class, milder: the paragraph opens immediately after the two-stage-chain paragraph and only reveals its scope in its final clause ("for the encrypted block"). First sentence should carry the scope word. | "The decrypted output of one block at 103 tokens has …" |
| 18 | `10_conclusion.tex:36` | "before mask and copy-repair overhead, **the ledger** projects a $60\%$ reduction" | Internal artifact name in reader-facing prose; a reader does not know what "the ledger" is. Also inconsistent with §6:151, which scopes the same 60% to "the total ciphertext–plaintext count" while §10 scopes it to "dense ciphertext–plaintext work". | "a projected reduction of roughly $60\%$ in the block's total ciphertext–plaintext count." |
| 19 | `07_results.tex:152-167` (Table 4) | "Share of wall" column: 0.4 + 0.1 + 0.0 + 98.7 + 0.0 + 0.9 | Column sums to 100.1%. Also, the ledger's own value for the evaluation share is 98.68 (`full.eval_share_of_wall`); the table prints 98.7 while the ledger scope string prints 98.68. | Either add a "column may not sum to 100 due to rounding" note or carry two decimals on the two largest rows. |
| 20 | `07_results.tex:181` | "the mean crossing costs $0.093$~s" | Ledger `full.client_per_crossing` is 0.0928 (three s.f.); the manuscript prints two. Everything else in §7.5 is quoted at ledger precision. | `0.0928`~s, or set the rounding convention in the ledger note. |
| 21 | `evidence/optimizations.yaml:152-165` (`net_full_pass`) | "The twelve-block evaluation is built and ready but **has not been run**, so this is a projection"; `to_range_human: "17 minutes to 2.2 hours"` | **Stale ledger entry that now contradicts `measurements.yaml: complete_model`.** Nothing in the manuscript draws on it (verified — no `2.2`, `17 min`, `26`-hour, `11.6`, or `7560` reaches the `.tex`), so this is not a manuscript defect. But the ledger is the paper's authority and it currently asserts two incompatible states. `telemetry.yaml` `meta` is likewise still `updated: 2026-08-08, total_wall_s: 663`. | Retire or rewrite `net_full_pass`; date-stamp `telemetry.yaml: meta` as single-block scope. |
| 22 | `evidence/optimizations.yaml:276-294` (`open_provenance`) | `prov.baseline_wall_time` and `prov.client_side_run` both still open | Per the standing rule, any claim depending on an open-provenance entry must be flagged. **None reaches the manuscript** — the 7560 s baseline, the 11.6x factor, the 14 GB client RAM and the 10% client share are all absent, and §6 explicitly declines the elapsed-time comparison. Recorded here so the next pass does not re-derive it. The block is not empty, so it is not yet submission-ready. | Close both, or restate. No manuscript change needed today. |

## Contamination check — contended per-block timings

**Clean.** None of `block_timing` (663, 652.545, 531.933, 120.612, 4.653, 1.956, 0.644, 0.045) or
`cost_shares` (80.2, 18.2, 18.5, 0.141, 0.161) appears anywhere in `sections/*.tex`. The single-block
walkthrough is drawn on only for quantities contention cannot affect — relative error (4.64e-9,
4.7e-9, 7.44e-8), host-memory stages (25,530 / 12,434 / 48,610 / 61,863 MiB), operation counts, and
depth — which is exactly what `AGENTS.md` permits. `fig_timeline` no longer plots the single-block
trace; `fig_waterfall` is deleted and de-registered.

## Do I accept the dedicated-node claim for the 2026-08-12 run?

**Yes, on balance — but two of the five recorded checks are weaker than their `[V]` tags imply, and
one denominator is wrong.**

What I confirmed independently, and what strengthens the claim beyond what `telemetry.yaml` records:

- `slurm.out` header: all three A100-PCIE-40GB devices report `0 MiB, 0 %` at job start. Supports
  `clean.node_empty_at_allocation` — but note this is *GPU-only* evidence and says nothing about CPU
  co-tenants at t=0.
- `clean.other_gpus_are_our_own_contexts` is correct and is under-argued in the ledger. The decisive
  arithmetic is missing from it: `gpu_others_mem_mib` is 0 for the first four samples and exactly 850
  for the remaining 649; it switches to 850 at t=42 s, the *same sample* at which `gpu0_mem_used_mib`
  first becomes 425, and `run.log` at `[42 s]` shows this process enumerating GPU 0, GPU 1 and GPU 2.
  850 = 2 x 425 = twice this process's own initial context footprint. That is conclusive, and it
  should be written into the check.
- The run used `-p gpu-a100` (whole-node partition), not the `gpu-a100-small` third-of-a-node
  partition that the contended single-block runs used. This is the single best structural argument
  for the dedicated-node claim and it is not recorded in `telemetry.yaml` at all.

Where I disagree with the recorded reasoning:

- `clean.no_cpu_oversubscription` says load "peaks at 29.1 **against the 32 allocated cores** and
  never exceeds the core count". `max(host_load1)` = 29.05 over 653 samples, so the number is right.
  The denominator is not: 32 is `OMP_NUM_THREADS` / `-c 32`, and the LS6 `gpu-a100` node carries far
  more cores than 32. So "never exceeds the core count" is comparing against the wrong quantity. The
  *correct* and stronger reading is that node-wide runnable demand never exceeded 29.05 while our own
  process alone reached `cpu_cores_busy` = 26.8 — i.e. there is little room left for a co-tenant.
  Note also that `host_load1` is a one-minute decayed node-wide average and `cpu_cores_busy` is an
  instantaneous per-process figure, so the two cannot be differenced to bound co-tenant CPU. Restate
  the check without the 32-core denominator.
- The batch script requests **no** `--exclusive`. Node exclusivity therefore rests on site scheduling
  policy, which is not recorded in any committed artifact. A reviewer cannot verify it from what is in
  the repository. Record `SLURM_JOB_CPUS_PER_NODE` / `SLURM_JOB_NUM_NODES` / `scontrol show job` output
  in the next run, or state the reliance on policy.
- All five `checks` are tagged `[V]`. Four of them are *inferences from* measurements, not
  measurements: "no co-tenant was present", "that is this process's own contexts", "the client-side
  timings are not inflated". Under this project's own tag discipline those are `[A]` conclusions drawn
  from `[V]` observations. Only `clean.memory_flat_across_blocks` is a direct reading, and finding 6
  shows it is overstated.

Net: I would not withdraw the timing. I would rewrite the contention block to argue from the
partition and the 850 = 2 x 425 coincidence, drop the 32-core denominator, and re-tag the inferences.

## Verified clean — do not re-audit

Every complete-model value in `measurements.yaml: complete_model` was checked against the run JSON,
`slurm.out` and the telemetry CSV, and every derived row recomputes exactly:

- `wall` 6683 s — `slurm.out: wall_clock_s=6683` (the run JSON has no wall field; the ledger `source`
  correctly names both files).
- `encrypted_evaluation` 6594.748483198, `server_linear_algebra` 5638.561475474,
  `client_boundaries` 956.187007724, `blocks_evaluation` 6565.992935047, `head_evaluation`
  9.045895324, `refreshes` 19.709652827, `context_keygen_load` 3.944696831, `fixture_load`
  23.494043552, `encrypt_input` 1.264851686, `final_decrypt` 0.043833214 — all exact.
- Internal consistency: `blocks + head + refreshes` = 6594.748483198 = `encrypted_evaluation`, to the
  last digit. `server + client` = 6594.748483198 = `encrypted_evaluation`, exactly — the manuscript's
  "the two sum to the evaluation total without residue" is literally true, not approximately.
- Sum of the twelve `blocks[].evaluation_seconds` = 6565.992935047 = `blocks_evaluation`; sum of the
  twelve `blocks[].refresh_seconds` = 19.709652827 = `refreshes` (12 entries, so "the twelve
  refreshes" at §7.5:186 is right).
- Derived shares: 85.5008 / 14.4992 % of encrypted evaluation; 84.3717 / 14.3078 / 98.6795 % of wall.
  All round as banked.
- `uninstrumented_remainder`: instrumented sum 6623.495908481, 6683 − that = 59.50409 → 59.504. ✓
- `mean_per_block` 6565.992935047 / 12 = 547.16608. ✓ `client_per_crossing` 956.187007724 / 10299 =
  0.0928427. ✓ `wall_hours` 6683 / 3600 = 1.85639 → 1.86. ✓ Refresh share 19.7097 / 6594.7485 =
  0.2989% → "roughly 0.3%". ✓
- Correctness: `margin_rel_error` 8.55814015132133e-09 → 8.56e-09 ✓; `margin_abs_error`
  1.049791276130918e-07 → 1.05e-07 ✓; `label_matches: true` ✓; `tol: 0.04` ✓.
- Protocol: `round_trips` 10299 ✓, `nonlinearity_round_trips` 10286 ✓,
  `inter_block_full_hidden_refreshes` 11 ✓, `logical_boundary_instances` 1551081 ✓. Crossing
  decomposition closes: 10286 + 11 + 1 (last-token) + 1 (head SiLU) = 10299. §7.4's "11 …
  refreshes between consecutive blocks and one further refresh of the final token into the head" is
  exactly right.
- Circuit: `bootstraps` 0 ✓, `multiplicative_depth` 13 ✓, `block_matrix_products` 1872 ✓,
  ct–ct 18076 ✓, ct–pt 2133839 ✓, `packed_output_level` 6 for all twelve blocks ✓,
  `intermediate_decrypt_attempts` 0 ✓, `evaluator_has_private_key` false ✓.
- Memory from the CSV: `max(gpu0_mem_used_mib)` = 9839 ✓; `max(target_vmhwm_mib)` = 50930 ✓
  (`max(target_rss_mib)` = 50928 — the ledger correctly sources the high-water mark, and the timeline
  figure correctly plots RSS).
- Telemetry statistics: `median(gpu0_util_pct)` = 32 ✓, `max` = 51 ✓ (see finding 5),
  `max(host_load1)` = 29.05 → "29.1" ✓, 653 samples ✓, `gpu_others_mem_mib` = 850 in 649 of 653 ✓.

Also checked and correct:

- Token derivations: 600/6 + 3 = 103 ✓; 103/8 → 13 groups ✓; 13x14/2 = 91 tiles ✓; 10,500/6 + 5 =
  1,755 ✓; 1,755/8 → 220 groups ✓; 220x221/2 = 24,310 ✓. Core-promoter 70 bp → 12 + 1 = 13 is a
  round-up of 11.67, consistent with the ledger; 300/6 + 1 = 51 ✓; 400/6 → 67 + 1 = 68 ✓.
- Single-block operation schedule as printed in Table 2, §5 and §6: 156, 1,236, 177,734, 1,506,
  8,173, 8,776, 857, 129,162, 91, 727, 159,744 — all match the ledger.
- Boundary decomposition: 128,544 / 129,162 = 99.52% → "99.5%" ✓; 1,551,081 / 10,299 = 150.6, so
  "understates the nonlinear work by more than two orders of magnitude" ✓.
- ct–ct attention share: (727 + 727) / 1,506 = 96.5% → the ledger's 97% ✓.
- Claim discipline on the three standing traps: **7.92x** is called a "structural reduction" and an
  "operation count rather than latency" at both appearances (`05:211-212`, `07:72-75`) — never a
  speedup. **Depth 13** is "the smallest demonstrated passing depth" (`05:241-242`) — never faster.
  **Twelve-times-a-block-time** does not appear at all; no projection is presented as a measurement.
- Scope words on the four bans: no encrypted task-accuracy claim (§8:13-15 states the single-input
  check explicitly); no end-to-end claim in the body; the in-process boundary is stated at §8:16-22
  and §7.8:245-247; model confidentiality is conceded as operational at §3 "What is not claimed" and
  §8:29-39.
- Plaintext baseline table: 0.9124 / 0.916 / 22,604 / 0.680 / 0.897 / 0.831 / 0.562 / 0.753 against
  0.9151 / 0.69 / 0.87 / 0.85 / 0.62, with the same-model-not-same-split footnote — all match
  `plaintext_baseline` including the corrected DNAGPT reference.
- Figures: `fig_cost_split` "other" segment = 3.945 + 1.265 + 0.044 + 23.494 + 59.504 = 88.252, and
  5638.561 + 956.187 + 88.252 = 6683.0 exactly. `fig_scaling` marks only the genomic-signal bar as
  measured and hatches the rest "circuit size only, not timed" — correct discipline.
