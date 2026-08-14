# Post-edit bibliography and citation review — 2026-08-14

## Verdict

No citation blocker remains. The manuscript now has a credible literature position: it does not
claim to invent client assistance, treats THOR and MOAI as uncontrolled cross-system context, and
distinguishes model-preserving evaluation from model and circuit redesign. One load-bearing privacy
sentence should still be narrowed, and two source-attribution issues should be cleaned up before
submission.

The mechanical inventory is complete: the manuscript cites 30 keys, `refs.bib` defines the same 30
keys, and there are no missing or unused entries. `paper-docs/scripts/build.sh lint` passes all four
checks, including citation lint.

## Blockers

None.

## High-priority finding

### The DNA-embedding source does not establish equivalence to the underlying sequence

- **Location:** `paper-docs/manuscript/source/sections/03_scenario_threat_model.tex:124`
- **Current claim:** “The embedded vectors that cross the boundary are not a weaker secret than the
  sequence.”
- **Evidence:** Ouaari, Kreuer, and Pfeifer evaluate per-token embeddings from DNABERT-2, Evo 2, and
  Nucleotide Transformer v2, not DNAGPT. They report near-perfect reconstruction from per-token
  embeddings across those evaluated models. That supports treating DNA-model embeddings as
  sensitive and encrypting them; it does not establish information-theoretic equivalence to the raw
  sequence, nor does it directly test the embedded vectors used here.
- **Recommendation:** replace the equivalence claim with the narrower statement that per-token DNA
  embeddings remain sensitive because high-fidelity reconstruction has been demonstrated for other
  DNA foundation models. The introduction's wording at
  `paper-docs/manuscript/source/sections/01_introduction.tex:20-22` already has the right scope.
- **Primary record:** [arXiv 2603.06950](https://arxiv.org/abs/2603.06950).

## Medium-priority findings

### Software citations identify Concrete ML and TFHE-rs but do not support the comparative selection result

- **Location:** `paper-docs/manuscript/source/sections/02_background.tex:73-76`
- **Issue:** the two repository citations establish the software projects and TFHE-rs's
  Boolean/integer orientation. They do not substantiate this project's conclusion that available GPU
  acceleration and ciphertext expansion were less suitable for the matrix-heavy graph. That is a
  project-specific engineering observation.
- **Recommendation:** separate the sourced software description from the study's own backend
  evaluation, or cite the exact evaluated hybrid-model/API documentation and describe the observed
  comparison in the methods. No new performance number is needed.

### The MOAI author metadata drops the hyphen in Kwok-Yan Lam

- **Location:** `paper-docs/manuscript/source/refs.bib:281`
- **Issue:** the ICLR 2026 paper and OpenReview record list **Kwok-Yan Lam**; the entry has
  `Lam, Kwok Yan`. The rest of the MOAI record is correct: eight authors, ICLR 2026, title, and
  OpenReview identifier.
- **Recommendation:** use `Lam, Kwok-Yan`.
- **Primary record:** [OpenReview / ICLR 2026](https://openreview.net/forum?id=qJn4HtTzhH).

## Focused verification results

### THOR comparison scope — verified and now appropriately qualified

THOR reports a 12-layer, 768-wide BERT-base encoder at 128 tokens, ring degree (2^{16}), 13
levels, nominal 128-bit security, one A100 GPU, and a 602.26-second total. The manuscript rounds
that value to 602 seconds and now labels the comparison as cross-paper and uncontrolled at
`paper-docs/manuscript/source/sections/07_results.tex:295-302` and
`paper-docs/manuscript/source/sections/09_related_work.tex:36-51`. It no longer claims an identical
hardware/software setup or treats the comparison as an estimate of the effect of client assistance.
The THOR title, authors, CCS 2025 venue, page range, DOI, and ePrint link are correct in
`refs.bib:162-169`. Primary record: [IACR ePrint 2024/1881](https://eprint.iacr.org/2024/1881).

### MOAI result scope — verified

MOAI reports 283.95 seconds against THOR's 602.26 seconds in the same comparison environment and
describes this as a 52.8% reduction. Its separate 141.3-second value is amortized over 256 inputs;
the manuscript does not conflate that experiment with the THOR comparison. The claims about
consistent packing and rotation-free Softmax and LayerNorm at
`paper-docs/manuscript/source/sections/09_related_work.tex:45-47,95-96` are supported by the ICLR
paper.

### GSR corpus and reference wording — verified

The local GSR result uses all 22,604 examples and therefore includes examples used to train the
released head. DNAGPT's 0.9151 reference is the 0.1B M-variant PAS(AATAAA) result on its held-out
quarter. The manuscript now states the local result as release/pipeline fidelity rather than
held-out performance at `paper-docs/manuscript/source/sections/04_plaintext_baseline.tex:16-21`,
and the table footnote and discussion retain the split mismatch at lines 55-77. The local and cited
models are both the 0.1B M variant, so “same-model but not same-split” is supportable. Primary record:
[DNAGPT](https://arxiv.org/abs/2307.05628).

### Newly added related work — verified

- **THE-X:** title, nine authors, Findings of ACL 2022 venue, pages, DOI, and URL are correct in
  `refs.bib:298-305`. The paper incorporates the user device to decrypt, evaluate ReLU-based
  nonlinear steps, and re-encrypt, so the scoped predecessor statement at
  `sections/09_related_work.tex:23-24` is supportable. It should not be described as preserving all
  original nonlinear formulas. Primary: [ACL Anthology](https://aclanthology.org/2022.findings-acl.277/).
- **Powerformer:** title, authors, ACL 2025 venue, pages, DOI, and URL are correct in
  `refs.bib:309-316`. It replaces or approximates transformer components with HE-suitable functions,
  supporting the model-modifying contrast at `sections/09_related_work.tex:56-57`. Primary:
  [ACL Anthology](https://aclanthology.org/2025.acl-long.543/).
- **ELLMo:** title, eight authors, year, and ePrint identifier are correct in `refs.bib:320-325`,
  including the revised 2026 record. Its packing-aware matrix multiplication and depth-reducing
  nonlinear designs support the claims at `sections/09_related_work.tex:58-60,97-98`. Primary:
  [IACR ePrint 2026/198](https://eprint.iacr.org/2026/198).
- **Li--Micciancio:** authors, EUROCRYPT 2021 venue, pages, DOI, and ePrint URL are correct in
  `refs.bib:329-336`. The security paper supports caution when approximate-decryption results become
  available to an adversary. The manuscript carefully notes that this protocol returns a fresh
  ciphertext rather than a decoded value and that malicious deviations are outside its semi-honest
  model at `sections/08_limitations.tex:48-57`; it does not claim the measured numerical headroom is
  a flooding parameter. Primary: [IACR ePrint 2020/1533](https://eprint.iacr.org/2020/1533).

## Other citation relationships

The remaining cited claims were rechecked against the primary-record audit in
`reviews/2026-08-14-paper-sources.md`. No new mismatch was found in the genomic task references,
CKKS/OpenFHE descriptions, private-transformer taxonomy, genomic HE lineage, or matrix-primitives
paragraph. The NEXUS sentence correctly warns that its headline per-input figure is batch-amortized;
the manuscript avoids quoting that figure as single-query latency. FIDESlib's OpenFHE
interoperability and GPU primitive claims are externally supported. The lack of ciphertext
serialization at `sections/09_related_work.tex:79-81` remains an observation about the evaluated
API, not a claim made by the FIDESlib paper.
