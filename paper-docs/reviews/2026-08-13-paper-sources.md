# Bibliography and citation audit — 2026-08-13

Scope: citations added since `reviews/2026-08-08-paper-sources.md` (§4 Table 1 Source column and
prose, §2 task-families and backends paragraphs), re-verification of the DNAGPT 0.9151 correction,
a claim-by-claim audit of §9, and a dangling/uncited sweep. Method: read-only, primary sources
fetched directly (arXiv/ar5iv, IACR ePrint PDFs, publisher DOI resolution). No files edited by the
auditing agent; this report was transcribed by the main session.

## 0. Headline finding

**THOR is a near-exact configuration match for this paper's complete-model run, and the manuscript
never states its numbers.** From the THOR paper (§7.1, §7.3.1, Table 6):

- BERT-base, **L = 12 layers, d = 768**, **n = 128 tokens**
- Ring **N = 2^16**, **13 multiplicative levels** before bootstrapping, **128-bit security**
- "All experiments were conducted on a **single NVIDIA A100 GPU**"
- **Total end-to-end 602.26 s (10.04 minutes)**; bootstrapping 337.86 s = 56.1% of runtime
- **Fully non-interactive** — no client boundary at all

Against this paper: 12 blocks + head, d = 768, 103 tokens, ring 65536 (= 2^16), depth 13, 128-bit,
one A100, **6683 s**, 10,299 client crossings. Same ring, same depth, same security level, same
accelerator, same layer count and hidden width, *fewer* tokens — and **11.1x slower, with an online
client required**.

Not a misattribution: §9's characterisation of THOR is accurate as far as it goes. It is an
attribution-fairness and completeness problem. A reader can finish §7's practicality verdict and
§9's THOR paragraph without learning that a prior non-interactive system runs a same-shape workload
on the same GPU in a sixth of an hour. Every reviewer in this area knows that number. MOAI (below)
reports a further 52.8% reduction against THOR, which sharpens it.

**In the paper's favour: no work was found performing encrypted inference on a genomic foundation
model.** The specific combination — released genomic transformer, all twelve blocks plus the
released task head, task-defined prompt length, exact nonlinearities at declared client boundaries,
decrypted label matched to a frozen plaintext oracle — is unclaimed as of this audit. The exposure
is latency, not priority.

## 1. Newly added citations

| Citation | Location | Verdict |
|---|---|---|
| `dnagpt2023` | Table 1 GSR row, §4 | **verified** — Table S2, DNAGPT-M, Human_PAS(AATAAA) 91.51; split 6:1.5:2.5 confirms "held-out quarter" |
| `dnabert2` | Table 1 x3, §4 | **verified with precision flag** (below) |
| `xpresso` | §2, §4 | **verified** — CNN over promoter sequence + 8 half-life features; DNAGPT states "from 0.59 to 0.62" |
| `deepgsr` | §2 | **verified** — 2D-CNN, two conv layers over a 598x64 trinucleotide matrix |
| `concreteml` + `tfhers` | §2 backends | **partly misused** (below) |

### DNABERT-2 precision flag

The three MCCs (69.37 / 86.77 / 84.99) are DNABERT-2's *own model rows*, not the best rows in that
table. DNABERT (3-mer) reports 70.92 core-promoter and 90.44 promoter-300; NT-2500M-multi reports
70.33 / 91.01 / 89.36. The sentence is literally true, but a column headed "Reference" invites the
reading "the benchmark's best result" — and this matters because 0.897 exceeds DNABERT-2's 86.77
while sitting below both DNABERT-3mer and NT. Repair without changing a number: say the values are
reported *for DNABERT-2 itself*.

Note: the GUE table is Table 6 in the ar5iv rendering and Table 4 in the 2026-08-08 audit. Do not
pin a table number in a `note` field.

### Backends sentence — three sub-claims, three support levels

1. "Boolean/integer-oriented backend" — **verified** (TFHE-rs tagline, verbatim).
2. "whose transformer protocol places nonlinearities at the client" — **true but unsourced by these
   entries**; it is documented on Concrete-ML's LLM-inference docs page, a different artifact from
   the GitHub repository `\cite{concreteml}` points at.
3. "Its available GPU acceleration and ciphertext expansion were less suitable" — **not supported by
   either citation.** Neither entry is a source for a comparative performance judgement.
   Concrete-ML's own documented figures are ~300 s/token CPU vs ~11 s/token GPU for GPT-2, a ~27x
   GPU speedup, which does not read as "less suitable GPU acceleration" to a reader who checks.
   Narrow the sentence to what the citations support, or frame the judgement as this project's own
   measured selection rather than a property of the library.

Also absent: the TFHE scheme's scholarly citation (Chillotti, Gama, Georgieva, Izabachene,
*TFHE: Fast Fully Homomorphic Encryption over the Torus*, J. Cryptology 33:34-91, 2020), verified
only through third-party reference lists — check directly before citing.

## 2. Reference metrics in the ledger

All GSR and GUE reference values verified against source. One misuse:

**`base.mrna_r2` — variant mismatch, unflagged.** DNAGPT reports 0.62 for the DNAGPT-**M**
(mammal-pretrained) backbone. `docs/tasks.md` records this work's mRNA evaluation as the released
`regression.pth`, fine-tuned from **`dna_gpt0.1b_h`** — the **H** variant. The row compares an
H-backbone head against an M-backbone number, and unlike the GSR row carries no caveat. The GSR
row's footnote sets the standard the mRNA row should meet.

Caveat on the finding: the auditing agent could not parse the DNAGPT PDF directly (size limit); the
M attribution rests on two independent ar5iv reads plus a search corroboration. The H-side of the
mismatch is confirmed directly from `docs/tasks.md`.

## 3. §9 Related work — claim-by-claim

**No mischaracterisations found.** Verified as accurate: `gazelle2018`, `iron2022`, `nimbus2024`,
`bumblebee2025`, `safhire2025`, `thor`, `idash2018`, `privategenomicqueries2017`,
`gpu-ckks-backend`, `aegis`, `heblas2025`.

Bibliographic corrections required:

- **`heblas2025`** — now published in the *Journal of Cryptology*, DOI `10.1007/s00145-026-09580-x`.
  Citing as a bare arXiv preprint is out of date.
- **`aegis`** — arXiv comments now read "Accepted at ICS 2026"; no longer a bare preprint.
- **`iron2022`** — add pages 15718–15731 (NeurIPS 35).
- **`nimbus2024`** — add pages 21572–21600 (NeurIPS 37).
- **`gpu-ckks-backend`** — 2026-08-08 open item now closed: poster paper at IEEE ISPASS 2025.

Recommended softening: §9 says the memory wall "establishes" a backend-specific conclusion. The
inference crosses libraries (THOR uses Liberate.FHE), so "indicates" or "supports" is the defensible
verb.

## 4. Dangling and uncited

**None in either direction.** All 22 keys used resolve to entries; all 22 entries are cited at least
once. §3, §5, §6, §7, §8, §10 carry no citations.

## 5. Missing coverage

Each was read before being listed.

1. **NEXUS** (Zhang et al., NDSS 2025) — first non-interactive secure transformer inference over
   RNS-CKKS. Reports BERT-based inference in 37.3 s with 164 MB bandwidth, GPU version 42.3x
   speedup. **Characterise carefully**: THOR reports NEXUS at ~2.7 h *latency* per prediction, so
   37.3 s is a batched/amortized figure — cite both readings or neither. The most conspicuous
   absence: it is the paper a reviewer names the moment they read that this protocol needs 10,299
   boundary crossings.
2. **BOLT** (Pang et al., IEEE S&P 2024) — the standard two-party baseline that Iron, BumbleBee,
   Nimbus, THOR and NEXUS all measure against. Flagged in the 2026-08-08 audit and still missing;
   one key added to §9.1's existing `\cite` list closes it.
3. **MOAI** (IACR ePrint 2025/991) — single packing convention across all layers, rotation-free
   Softmax and LayerNorm; removes 2,448 rotations from BERT-base and reports **52.8% less
   evaluation time than THOR**. Current non-interactive ceiling. Cite the ePrint only; ICLR 2026
   acceptance unconfirmed.
4. **Ouaari, Kreuer & Pfeifer**, *How Private Are DNA Embeddings?* (arXiv 2603.06950, Mar 2026) —
   per-token embeddings from DNA foundation models permit near-perfect sequence reconstruction
   across DNABERT-2, Evo 2, and Nucleotide Transformer v2. **Belongs in §1 and §3.** This paper's
   encrypted scope begins at embedded numeric token vectors; this is published evidence that those
   vectors are not safe to hand a provider in clear, converting a scope boundary the reader might
   read as convenient into a cited motivation. Also the strongest answer to "why not just send
   embeddings?"
5. **Zephyr** (IACR ePrint 2026/932) — grafting-based CKKS representation for 32-bit GPU execution;
   CCMM optimization merging overlapping rotation patterns in attention. Cite only if §6 engages
   with it, otherwise decoration.

**Explicitly not recommended without deeper reading** (abstract-level only): Cheddar, HEonGPU, ENSI,
EncFormer, *FHE on Llama 3*, *A Survey on Private Transformer Inference*, *Privacy-Preserving
Federated Inference for Genomic Analysis with HE*.

## 6. Open items

- Confirm "We report the results of DNAGPT-M in this task" against the DNAGPT paper text directly.
- Do not pin the DNABERT-2 GUE table number in any `note`.
- MOAI venue unconfirmed; cite ePrint.
- TFHE scheme paper verified only via third-party reference lists.
- **Terminology collision noticed outside remit:** §7 uses "non-interactive" in the plain-English
  sense while the paper uses it throughout as the term of art for the set-aside design. Word change
  needed; flagged for `paper-style`.
