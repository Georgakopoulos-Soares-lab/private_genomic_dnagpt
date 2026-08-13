# Structural review — 2026-08-13 (INCOMPLETE)

**Status: the `paper-structure` agent was terminated twice mid-run** — first by an organisation
monthly spend limit, then by a stalled API response. It had begun editing before dying, so this
record was written by the main session by inspecting the resulting file state, not by the agent.
It is a record of what landed, not a completed structural review.

## Edits that landed and were kept

1. **Section order changed: related work now precedes limitations.** `main.tex` inputs
   `09_related_work` before `08_limitations`. The rationale is sound and was one of the questions
   put to the agent: §9 now carries the competitive cost comparison, so a reader who reaches
   limitations without it forms an incomplete judgement. Filenames still read 08/09 while render
   order is 9-then-8; labels (`sec:related`, `sec:limitations`) are intact and all cross-references
   resolve.
2. **`\label{sec:related}` added to §9**, so limitations can cite it.
3. **New §8 paragraph on interaction as a standing requirement.** States that the protocol needs an
   available key holder at every one of the 10,299 boundaries, that a data owner going offline
   mid-evaluation cannot finish, that non-interactive designs carry no such dependency and report
   lower cost at comparable shape and identical parameters, and that this study does not claim
   client assistance is preferable on latency, throughput, or availability. This is the strongest
   honesty improvement of the session.
4. **Scope word added to §7 correctness**: the 4.64e-09 figure is now explicitly "for one block at
   the 103-token task prompt", so it cannot be read as a complete-model number.

## Regression caused by the interruption, since repaired

The agent rewrote §7's practicality verdict from an older copy and reverted an earlier fix, so
"confines it to non-interactive use" reappeared — colliding with the paper's use of
*non-interactive* as the term of art for the set-aside design. Restored to "batch and offline use".

## Completed by the main session

The cross-reference the agent was adding when it died: §7's practicality verdict now names the
602 s non-interactive comparison and points to §9, and states plainly that this study does not claim
client assistance as the cheaper route. The qualification list went from two items to three.

## Structural questions still UNANSWERED

These were put to the agent and it did not reach them. They remain open and are the reason this
review is marked incomplete.

- **(a) Does §1 still promise what §7 delivers?** §1 frames three candidate barriers — correctness,
  memory, cost. Two are now resolved and the third has a number that compares unfavourably with
  prior work. The contribution may need restating around exactness, verifiability, and the genomic
  target rather than around barrier-elimination.
- **(b) Is §6 proportionate?** It is a full section on an optimization campaign that can report no
  speedup figure and never will, because the pre-optimization baseline has no clean measurement.
  Candidate for contraction into §5 or §7.
- **(c) Does §5's non-interactive subsection still land?** It sets that design aside for a GPU memory
  wall. Now that §9 concedes non-interactive systems are faster at these exact parameters, the
  subsection risks reading as dismissing the approach that wins on cost. §9 already says the wall
  was backend- and parameter-specific; §5 should probably say it too, at the point of dismissal.
- **(d) Remaining order question:** whether §4 (plaintext reference) is correctly placed after §3
  and before §5.

## Verification at time of writing

`scripts/build.sh` clean on all four checks; PDF compiles with zero undefined references and zero
overfull boxes.

---

# Structural review — 2026-08-13, second pass (COMPLETE)

Re-run of the interrupted review above, after the complete-model result and the §9 competitive
comparison landed. This pass answers questions (a)–(e) left open above and makes the edits.

## 0. The tree did not contain the edits the record above claims

Before anything else. The section above lists four edits as "landed and kept" and two more as
"completed by the main session". **None of them was present in `manuscript/source/` at the start of
this pass.** Specifically:

- `main.tex` still input `08_limitations` before `09_related_work`.
- §9 had no `\label{sec:related}`.
- §8 had no interaction paragraph.
- §7's correctness paragraph had no scope word on `4.64e-9`.
- §7's practicality verdict named no competitive comparison.
- §7's closing sentence still read "confines it to non-interactive use" — the exact regression the
  record says was repaired.

The most likely explanation is the `.gitignore` merge in `149e9c8` / `6666839`: the review file was
committed and the `.tex` edits were lost. **Action for the next session: do not trust
`reviews/*.md` as a record of file state.** Every item above has now been re-applied. Mid-pass, the
main session applied its own repairs to `07_results.tex` concurrently, producing a duplicated
602 s comparison; that has been deduplicated.

## 1. Answers to the five questions

**(a) Does §1 still promise what §7 delivers? Partly, and the frame needed widening — not
replacing.** The three-barrier frame is mandated by the project objective and stays. What broke is
that §1 and the abstract sold *barrier elimination* as the contribution, and with THOR in the paper
that reads badly in two places: (i) "memory is a barrier that design removes" is a headline about a
barrier a published non-interactive system does not hit, and (ii) cost is now conceded rather than
merely reported. The contribution has been restated around **exactness, absence of bootstrap, and
oracle-checkable output** — not by removing the barrier frame, but by scoping the memory claim at
every appearance and adding an explicit non-claim of speed. Edits: abstract Conclusions, two Key
Points (memory bullet scoped, new positioning bullet), §1 new paragraph 5, §1 results paragraph,
§1 contribution bullet 4.

**(b) Is §6 proportionate? Yes, but only after being told what it is for.** Contraction was the
wrong call. §6 carries a genuine result that is not a speedup: the obvious host-side economy
(encode each weight diagonal once) is *fatal* in its obvious form — the unbounded cache was killed
by the OS at 61,863 MiB at the MLP stage — and bounding it to one live stage is what makes it
usable. That is an enabling finding, and it was buried in §6.3. It now opens the section. The
section is retitled **"Host-side systems work at fixed circuit"**, because "optimization" promises
a speed number the section cannot pay. Length is now proportionate: §5 ≈ 255 lines, §7 ≈ 285, §6
≈ 160, §4 ≈ 101. The plaintext-baseline section is *not* as long as the optimization section, so
the balance test passes.

**(c) Does §5's non-interactive subsection still land? It did not; it does now.** It set the design
aside for a memory wall while §9, eight pages later, conceded the wall was ours. A reader met the
dismissal before the concession. §5.2 now carries a forward reference to §9 at the point of
dismissal, saying that published non-interactive systems evaluate comparable encoders at these
parameters and that the subsection reports where *our* configuration stopped, not where the
approach stops. The same scoping was applied at every other appearance of the claim: abstract, Key
Points, §1, §7.3, §7.7.

**(d) Ordering.** §4 before §5 is **correct and unchanged** — §5's notation table depends on
`T = 103` and `G = 13`, which §4 derives, and the frozen reference must exist before the protocol
verified against it. §8/§9 **swapped**: related work now precedes limitations, so §8 can concede
the competitive comparison rather than introduce it. All labels intact; every `\ref` resolves.

**(e) Where should the THOR comparison live? In three places, not one.** §9 alone is too late — a
reader forms the practicality judgement in §7.8. It is now stated: in §1 (at full strength, before
our own numbers), in §7.8 (as the comparison that decides the verdict), and in §9 (developed). §8
then concedes the interaction dependency that follows from it.

## 2. Breaks in the argument, and what was done

Most severe first.

1. **The strongest objection was not stated at full strength.** `context/01` requires the
   local-inference objection to appear with every number conceded, before the answer. §3.2 stated
   it qualitatively with no numbers at all. It now opens with the concession in our own
   measurements: the data owner spends 956.187 s of its own CPU inside the protocol against a
   plaintext forward pass finishing in well under a second, and the weights in this study are
   public so no model-secrecy argument compels the arrangement either — *then* the answer. Same
   treatment applied to the competitive objection in §1.
2. **§6 claimed no clean timing artifacts exist, contradicting §7.** (Also evidence finding 12.)
   Rewritten: no dedicated-node measurement of the *pre-optimization configuration* exists, and a
   speedup against a contended baseline would not measure these changes.
3. **The memory headline outran its scope in five places.** Fixed at each: abstract, Key Points,
   §1, §7.3, §7.7 now all say the envelope belongs to the evaluated backend, packing, and
   parameters.
4. **§8 omitted the limitation §9 concedes.** Interaction was nowhere listed as a standing cost.
   New paragraph: a key holder must be available at all 10,299 boundaries, a data owner going
   offline cannot finish, no other party can complete a boundary, non-interactive designs carry no
   such dependency and report lower cost, and this study does not claim client assistance is
   preferable on latency, throughput, or availability.
5. **Scope words missing before headline numbers.** §7.1's `4.64e-9` now reads "for one block at
   the 103-token task prompt"; §8's now reads "the measured single-block error", with the
   twelfth-block hidden state quoted alongside. Abstract's "Complete encrypted DNAGPT inference is
   therefore feasible" now carries "over the released blocks and task head, beginning at embedded
   token vectors".
6. **Projections stated as exact figures.** (Evidence findings 14, 15.) "approximately 17,400",
   "approximately 52", "approximately 53,248", "roughly 60%" restored, plus an explicit "this is a
   derived projection for an unimplemented layout".
7. **"below 51%" vs "never exceeds 51%".** (Evidence finding 5.) Unified on "never exceeds" in all
   five places. Fig. 4's caption no longer says GPU memory is "flat throughout" — it now says the
   peak is reached during the first block and does not grow with block index.

## 3. Gaps a reviewer will ask about that the paper still cannot answer

1. **No measured plaintext forward-pass time.** §3.2 and §7.8 both compare against "well under a
   second on commodity hardware", and §3.2 now says the client does "orders of magnitude more
   work". Neither has a ledger row. This is the one number a reviewer can falsify on a laptop, and
   it costs no GPU time. **Bank a `[V]` row for DNAGPT-0.1b at T=103 on the client CPU class,
   including model load, and quote the derived ratio as `[A]`.** Until then the comparison is
   qualitative by necessity, not by choice.
2. **No ledger row for the THOR ratio.** §1, §7.8 and §10 say "roughly an order of magnitude"
   rather than 11x, because 6683/602 is not banked. Either bank it as an `[A]` derived row or leave
   the qualitative form; do not let a stray "11x" appear without one.
3. **NEXUS and BOLT.** Flagged by `paper-sources`. NEXUS is now cited in §9; BOLT is cited. NEXUS
   is the paper a reviewer names on reading "10,299 boundary crossings", and §9's handling of its
   amortized-vs-latency readings should be checked once more before submission.
4. **Client peak RAM is absent everywhere.** The deployment claim in §3 is "runs on a laptop", and
   the paper never gives a client memory figure because `client.peak_ram` is attested, not
   committed. A reviewer will ask. It needs no GPU.
5. **`clean.no_cpu_oversubscription` rests on a 32-core denominator that is not the node's core
   count**, and the batch script requests no `--exclusive`. The dedicated-node claim carries every
   timing in the paper. This is an evidence-side repair, but it is load-bearing for §7 and §8.

## 4. Does new evidence still land without a rewrite?

- **A second complete-model run (n≥2).** Lands cleanly. The n=1 qualifier is now concentrated in
  three places — §7.8's two-item qualification list, §8's opening paragraph, and §10 — each a
  single sentence to rewrite. Table 4 absorbs a second column or a second row unchanged.
- **A dedicated-node pre-optimization baseline.** Lands cleanly. §6's third paragraph is written so
  that one sentence changes and the table gains one column. This was *not* true of the previous
  wording, which asserted the artifacts do not exist.
- **A measured plaintext forward-pass time.** Lands in §3.2 and §7.8 as a number replacing a
  qualitative phrase, in the sentence already built for it.
- **Per-task encrypted timings.** Would land in Fig. 5 as measured bars replacing hatched ones; the
  figure already distinguishes the two. No prose unwinding needed.

## 5. What is working — do not restructure this away

- **The two-verdict architecture** (§7.7 / §7.8 with their own labels) is the paper's best
  structural decision and now carries more weight, not less: feasibility affirmative, practicality
  negative in two distinct senses. Do not merge them.
- **The two-stage verification chain** (§4 → §7.1) is what makes "nine significant figures" mean
  anything, and it is stated once, early, and referenced rather than repeated.
- **§7.1's error-accumulation paragraph** — conceding that error grows two orders of magnitude
  across depth and then showing the margin is still nine orders inside tolerance — is exactly the
  right move and is stronger than claiming error did not grow.
- **The invariant operation schedule as a run-time assertion** rather than an argument. It is what
  licenses §6's attribution to systems work, and it is now stated in §6's second paragraph, where
  the claim is made.
- **§3's "What is not claimed"** — conceding model confidentiality before a reviewer raises it.

## 6. Not done, needs a decision

- **Moving §9 earlier still, to follow §2.** There is a case: §2.3's design-space argument is
  populated by exactly the systems §9 describes, and a reader currently meets THOR's parameters in
  §1 with no context. Against: §9's later subsections reference §6's remaining targets, and the
  move touches four cross-references. Left alone; the §8/§9 swap captures most of the benefit.
- **`AGENTS.md` line 42 is stale**: "The twelve-block evaluation is built and ready to launch." It
  has run. Not edited — outside the `.tex` remit.
- **Ledger `net_full_pass` is stale** and contradicts `measurements.yaml: complete_model`
  (evidence finding 21). Nothing in the manuscript draws on it. Not edited — ledger is read-only
  for this agent.

## 7. Verification

`build.sh lint` **could not be executed** — this agent had no shell tool in its environment.
Preconditions were verified by inspection instead, and every one holds:

- **Numbers.** The only numeric literal introduced that was not already elsewhere in the manuscript
  is `61{,}863` (§6), present in `optimizations.yaml` under
  `opt.bounded_host_memory.enabling_not_merely_faster`. All others (602, 6683, 956.187, 10,299,
  61,863, 2^16, 13, 128, 768, 12, 103, 0.1, 60, 51, 100) already appear in sections that lint
  clean, or are in `ALWAYS_OK`.
- **Citations.** `\cite{thor}` and `\cite{moai2025}` added in §1 and §7; both resolve to
  `refs.bib` entries. Zero `UNVERIFIED` markers.
- **Structure.** `\label{sec:related}` added; all `\ref` targets resolve; no label removed or
  renamed; no `\includegraphics` target changed.
- **Banned terms.** No new occurrence of any pattern in `check_numbers.BANNED`.

**Run `paper-docs/scripts/build.sh lint` before committing** to confirm, and rebuild the PDF: the
§8/§9 swap changes section numbering, so every cross-reference renumbers.
