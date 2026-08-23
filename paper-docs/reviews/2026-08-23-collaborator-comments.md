# 2026-08-23 — disposition of collaborator comments (`comments.md`)

Source: `comments.md` at the repository root, format `"quoted text" > comment`. Each row records
what the manuscript now says and where. "Already applied" means the change was present before this
pass (from the previous comment round) and was re-checked, not re-edited.

| # | Comment | Disposition |
|---|---|---|
| 1 | Abstract hard to follow | **Applied.** `00_frontmatter.tex` Results and Conclusions split into shorter declarative sentences; no number, qualifier, or scope word removed. |
| 2 | "We validate DNAGPT…" rephrase | Already applied — now "We first reproduce DNAGPT on three genomic task families and freeze its per-example plaintext predictions". |
| 3 | Key Points too technically loaded | **Applied.** All four bullets rewritten in plain terms ("Nothing in the released model is changed to make this possible", "the circuit only has to be deep enough for the longest stretch between two returns"). Numbers and tags unchanged. |
| 4 | Add "6.67 orders of magnitude below the 4×10⁻² tolerance" | Already applied (abstract + Key Points). |
| 5 | Validation-chain sentence too technical | Already applied — Key Point 2 is now "checked against a reference that is itself checked". |
| 6 | "not interactive" is the wrong word | Already applied; strengthened in Key Point 4: the protocol is interactive by design, what rules out interactive use is speed. |
| 7 | Graphical abstract needs legend + text mention | **Applied.** Caption already existed; `01_introduction.tex` now cites Figure~\ref{fig:abstract}. |
| 8 | Provider does more than linear algebra | Already applied — "the encrypted model arithmetic between client boundaries". |
| 9 | DNAGPT is not human-only | Already applied — "the downstream tasks evaluated here are defined over human sequence". |
| 10 | Embedding-inversion sentence replacement | Already applied. |
| 11 | "DNAGPT is public and can be run locally…" + stronger reasoning | Already applied (introduction, paragraph 2). |
| 12 | "That choice buys exactness and depth…" — remove | **Applied.** Paragraph deleted from `01_introduction.tex`; its content survives in Sections 5, 7, and 9. |
| 13 | "We therefore ask whether…" replacement | Already applied. |
| 14 | "The main cryptographic difficulty…" replacement | Already applied. |
| 15 | "Correctness" → "Numerical agreement" | Already applied. |
| 16 | Accelerator-memory sentence: rephrase | Already applied (split into three sentences in the introduction). |
| 17 | Add "graph" to "Complete-model feasibility" | Already applied in contributions; **also applied** in the conclusion this pass. |
| 18 | "Independently checked numerical evaluation." | Already applied. |
| 19 | Attention is not a nonlinear function | Already applied — `02_background.tex`: "attention introduces products between two data-dependent quantities and normalizes them with a softmax". |
| 20 | "The release includes fine-tuned heads…" — needed? | Kept, in compressed form (`02_background.tex`, three sentences). It is load-bearing: it explains why the understanding benchmark uses a locally fine-tuned head. |
| 21 | "The protocol instead addresses a served-model deployment…" — needed? | **Applied.** Redundant lead sentence dropped; the paragraph now opens "Where the client may not hold the weights…". |
| 22 | "asserts" → "validates" | **Applied** in the invariant, and for consistency in `05_protocol.tex`, `07_results.tex`, and the introduction. |
| 23 | "Transcript equality…" too strong | **Applied.** Now: the boundary schedule is fixed by public parameters rather than by the private input, and no separate leakage experiment compares transcripts across inputs of the same length. |
| 24 | Party-allocation figure needs number + text mention | **Applied.** `05_protocol.tex` now cites Figure~\ref{fig:architecture}. |
| 25–28 | Bounded cache / GPU trace / cost split / prompt-length figures should be panels of a larger figure, with text mentions | **Applied.** The four standalone figures are now panels (a)–(e) of one full-width figure, `fig_systems.pdf` (Figure 4, Results). `figures.py` refactored into `_panel_*` functions plus `fig_systems`; the four old PDFs are deleted. Panels are cited in text: (a) and (b) in the memory subsection and the practicality verdict, (c) in the cost subsection, (d) in Section 6.3, (e) in the sequence-length subsection. |
| 29 | 51 % GPU maximum does not establish the bottleneck | **Applied** in Results, the introduction, the conclusion, and the figure caption: "this does not identify the limiting resource". |
| 30 | Delete "These are uncontrolled cross-system comparisons." | **Applied** (`09_related_work.tex`); the following sentence carries the same content without the label. |
| 31 | "no contention detected" → "no evidence of resource contention" | **Applied** in `08_limitations.tex` (with the trailing clause dropped as requested) and, for consistency, in Results and the cost-table caption. |
| 32 | "no party other than the data owner can complete a boundary" — accurate? | **Applied.** Now "only a holder of the secret key can complete a boundary". |
| 33 | "Non-interactive designs carry no such dependency" too broad | **Applied.** Now "A non-interactive evaluation does not require the key holder to be available while the computation runs." |
| 34 | 9839 MiB — is that the peak? | **Applied.** Abstract, Key Points, and conclusion now say "peaks at 9839 MiB" (and, in the conclusion, "holds that value across the blocks"), matching the Results text. |
| 35 | "forecloses" → "does not enable" | **Applied** in the conclusion and the introduction. |
| 36 | Conclusion: add "graph"; "rather than an observed correctness failure" | **Applied.** |
| 37 | Future work: repeated timings belong up front; "should" → "could"; thread-safe-context wording | **Applied.** The paragraph now opens with repeated execution and confidence intervals, states that the present run supports none of the three, uses "could" for systems work, and adopts "would remove one barrier to concurrent boundary processing". |
| 38 | Repository needs a license | **Applied.** MIT (`LICENSE` at the repository root, chosen by the corresponding author). The code-availability statement now names it and notes that upstream model code, weights, and datasets keep their own licenses. |
| 39 | Check references for hallucinations | **Checked.** Re-verified against primary records: THOR (ePrint 2024/1881), Safhire (arXiv 2509.01253), ELLMo (ePrint 2026/198), MOAI (ePrint 2025/991 / ICLR), NEXUS, AEGIS (arXiv 2604.03425), HE-BLAS (arXiv 2503.16080), DNA-embedding inversion (arXiv 2603.06950), and the iDASH 2024 winning solution. One error found and fixed (row 40). Remaining entries carry earlier verification marks and were not re-fetched. |
| 40 | Yang et al. 2024 author list is wrong | **Applied.** ePrint 2024/1851 lists ten authors; the entry dropped the first four. Corrected to: Jingwei Chen, Linhan Yang, Chen Yang, Shuai Wang, Rui Li, Weijie Miao, Wenyuan Wu, Li Yang, Kang Wu, Lizhong Dai. |

## Second round: "package the graphs into fewer figures, make it more compact, remove repetitions"

Figures: 7 floats → 4 (graphical abstract, party allocation, packing, and the new five-panel
systems figure).

Compactness: 20 pages → 19; body prose 12,023 words. The bibliography is set at `\footnotesize`,
which removes a nine-word orphan page.

Repetitions removed (each fact now stated once, in the section that owns it):

| Repeated fact | Was in | Kept in |
|---|---|---|
| Three-link float32/float64/encrypted validation chain | §1, §4, §7.2 (three full statements) | §4 in full; §1 and §7.2 one sentence each |
| Encrypted-vs-float32 margin difference $1.61\times10^{-6}$ | §4 and §7.2 | §7.2 |
| Attention = 818 of 857 client calls, 818 of 896 objects | §5, §6.5, §7.3 | §5; §6.5 and §7.3 refer to it |
| "The implementation validates the schedule / counts / depth" | §5, §6, §7.3 | §5; §6 and §7.3 refer to it |
| Stage host memory 25,530 / 12,434 / 48,610 MiB | §6.3 and §7.4 | §6.3 (and the figure) |
| Two-block composition error $8.48\times10^{-11}$ | §5.7 and §7.5 | §7.5 (and Table 7) |
| Depth 13 / 9839 MiB / no bootstrap | abstract, Key Points, §1, §7.4, §7.5, §7.8, §10 | abstract, Key Points, §7.4–§7.5; §1, §7.8, §10 state the property without re-quoting the numbers |
| 600 bp → 100 + 3 tokens derivation | §4.1 and §7.7 | §4.1 |
| THOR's configuration and the uncontrolled-comparison caveat | §7.9 and §9 | §9 in full; §7.9 cites it |
| "single sample, not a mean or service rate" | §7.1, Table 8 caption, §7.9, §8, §10 | §7.1 (short), Table 8 caption, §8 |
| Released artifact is a reproducible surrogate for a served model | §1 and §3.1 | §1; §3.1 refers to it |
| Interaction dependency vs non-interactive designs | §8 (twice, with the §9 pointer) | §8, once |

Prose also tightened without cutting content in §2.3–2.4, §3.1–3.2, §5.2, §6.1–6.4, and §9.
Tables, algorithms, and every measured number are unchanged.

## Verification

`paper-docs/scripts/build.sh` — figures, PDF, lint — clean on all four checks (banned terms,
numbers against the evidence ledger, citations, structure). No new number entered the manuscript;
every value in the new figure is read from `evidence/*.yaml` as before.
