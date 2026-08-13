# Bibliography and citation audit — 2026-08-08

Scope: verify `paper-docs/evidence/measurements.yaml` baseline reference metrics, verify all
12 `refs.bib` entries, locate missing Section 9 (related work) literature, flag scoop risk.
Method: read-only agent audit (`paper-sources`), primary sources fetched directly where
possible. No `refs.bib` edits made — corrected BibTeX below is for the author to paste in.

State at audit time: no `\cite` commands exist anywhere in the manuscript yet (confirmed by
grep). Every `refs.bib` entry is currently unused; Section 9 is TODO placeholders only.

## 1. Baseline reference metrics — highest priority, checked first

| Metric | Verdict | Detail |
|---|---|---|
| DeepGSR accuracy ~0.916 | **WRONG — mislabeled source** | See below. |
| DNABERT-2 MCC ~0.69 core promoter | Verified | Table 4, DNABERT-2 model, human core promoter (all), MCC 69.37 → 0.69. Same split (47,356/5,920/5,920) as `docs/data_provenance.md`. |
| DNABERT-2 MCC ~0.87 300bp promoter | Verified | Table 4, "Promoter detection (all)" 300bp, MCC 86.77 → 0.87. Same split. Our 0.897 genuinely exceeds this — already stated correctly in `docs/tasks.md`. |
| DNABERT-2 MCC ~0.85 splice site | Verified | Table 4, "Splice site prediction (Reconstruct)", MCC 84.99 → 0.85. Same dataset (36,496/4,562/4,562, human, reconstructed/adversarial-negative variant). |
| DNAGPT r² ~0.62 mRNA abundance | Verified, correctly attributed | DNAGPT's own paper: "DNAGPT outperformed Xpresso from 0.59 to 0.62." `reference_name: "DNAGPT"` is correct as written. |

### DeepGSR finding — invalidates the current baseline row, must fix before submission

Read DeepGSR's own paper directly (Kalkatawi et al., *Bioinformatics* 35(7):1125–1132, 2019,
DOI 10.1093/bioinformatics/bty752), including Table 2 and all organism/signal rows. DeepGSR's
own reported result for human AATAAA polyadenylation-signal recognition: **13.06% error rate
= 86.94% accuracy.** No value near 0.916 exists anywhere in that paper.

The `0.916` currently in `measurements.yaml` matches **DNAGPT's own self-reported number**
(Table S2, 0.1B "M" variant — the class this repo evaluates): accuracy = F1 = 91.51% for human
PAS(AATAAA). It has been misattributed to DeepGSR.

Second, independent problem: DNAGPT's 91.51% was measured on a **held-out test split only**
(train:val:test = 6:1.5:2.5, ~5,652 of 22,604 examples). This repo evaluates the **full
22,604-example balanced set**. So even the corrected DNAGPT comparison is not same-split.

**Fix — pick one:**
- (A) Relabel `reference_name: "DNAGPT"`, `reference: 0.9151`, add a scope note that DNAGPT's
  number is a ~25% held-out split, not the full set we evaluate.
- (B) If a true DeepGSR comparison is wanted: `reference: 0.8694, reference_name: "DeepGSR"` —
  but then the surrounding prose changes meaning: "matches within 0.4pt" becomes "exceeds by
  ~4.3pt," a materially different and arguably stronger claim.

This affects `tab:baseline` and Figure `fig_baseline` directly — both currently imply a close
match to DeepGSR that doesn't exist in DeepGSR's own paper.

## 2. Corrected BibTeX — 12 entries

All titles/authors/venues below verified against live records (arXiv abstract pages,
publisher DOI pages, ACM/dblp/Semantic Scholar cross-reference where direct access 403'd).
Paste-ready; `refs.bib` was not edited.

### dnagpt2023
Current title "Multiple" is stale — arXiv v3 (Aug 2023) retitled to "Versatile."
```bibtex
@article{dnagpt2023,
  title   = {DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks},
  author  = {Zhang, Daoan and Zhang, Weitong and Zhao, Yu and Zhang, Jianguo and He, Bing and Qin, Chenchen and Yao, Jianhua},
  journal = {arXiv preprint arXiv:2307.05628},
  year    = {2023},
  url     = {https://arxiv.org/abs/2307.05628}
}
```

### dnabert2
Also the correct citation for GUE — GUE is introduced in this same paper; no separate `gue`
entry needed.
```bibtex
@inproceedings{dnabert2,
  title     = {DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author    = {Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana V. and Liu, Han},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2306.15006},
  archivePrefix = {arXiv},
  note      = {Source of the GUE benchmark and the promoter/splice-site reference metrics (Table 4)}
}
```

### deepgsr
Bibliographic details verified correct. The *use* of this entry's number in the evidence
ledger is not — see §1.
```bibtex
@article{deepgsr,
  title   = {DeepGSR: an optimized deep-learning structure for the recognition of genomic signals and regions},
  author  = {Kalkatawi, Manal and Magana-Mora, Arturo and Jankovic, Boris and Bajic, Vladimir B.},
  journal = {Bioinformatics},
  volume  = {35},
  number  = {7},
  pages   = {1125--1132},
  year    = {2019},
  doi     = {10.1093/bioinformatics/bty752}
}
```

### xpresso
Current title in `refs.bib` ("Deep Learning of the Regulatory Grammar of Yeast and Human
Gene Expression") does not match any Agarwal/Shendure paper — appears conflated with an
unrelated Cuperus et al. yeast-UTR paper from a different group. Fabricated/misattributed
title; must not survive into the PDF as-is.
```bibtex
@article{xpresso,
  title   = {Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks},
  author  = {Agarwal, Vikram and Shendure, Jay},
  journal = {Cell Reports},
  volume  = {31},
  number  = {7},
  pages   = {107663},
  year    = {2020},
  doi     = {10.1016/j.celrep.2020.107663},
  note    = {Source of the mRNA abundance split; r^2=0.62 reported by DNAGPT, r^2=0.59 for Xpresso itself, same comparison}
}
```

### ckks2017
Author list/year already correct; only venue detail (LNCS vol, pages) was missing.
```bibtex
@inproceedings{ckks2017,
  title     = {Homomorphic Encryption for Arithmetic of Approximate Numbers},
  author    = {Cheon, Jung Hee and Kim, Andrey and Kim, Miran and Song, Yongsoo},
  booktitle = {Advances in Cryptology -- ASIACRYPT 2017},
  series    = {Lecture Notes in Computer Science},
  volume    = {10624},
  pages     = {409--437},
  year      = {2017},
  doi       = {10.1007/978-3-319-70694-8_15}
}
```

### openfhe
```bibtex
@inproceedings{openfhe,
  title     = {{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  author    = {Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and Liu, Zeyu and Micciancio, Daniele and Quah, Ian and Polyakov, Yuriy and Saraswathy, R. V. and Rohloff, Kurt and Saylor, Jonathan and Suponitsky, Dmitriy and Triplett, Matthew and Vaikuntanathan, Vinod and Zucca, Vincent},
  booktitle = {Proceedings of the 10th Workshop on Encrypted Computing \& Applied Homomorphic Cryptography (WAHC'22)},
  year      = {2022},
  note      = {Also IACR ePrint 2022/915}
}
```

### fideslib
```bibtex
@misc{fideslib,
  title  = {{FIDESlib}: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs},
  author = {Agull\'{o}-Domingo, Carlos and Vera-L\'{o}pez, \'{O}scar and Guzelhan, Seyda and Daksha, Lohit and El Jerari, Aymane and Shivdikar, Kaustubh and Agrawal, Rashmi and Kaeli, David and Joshi, Ajay and Abell\'{a}n, Jos\'{e} L.},
  year   = {2025},
  eprint = {2507.04775},
  archivePrefix = {arXiv},
  note   = {Poster paper, IEEE ISPASS 2025 (poster status corroborated but not confirmed via primary ACM/IEEE source — do one direct check before submission)}
}
```

### concreteml / tfhers — split into two entries
Claim is about the TFHE-rs backend specifically (GPU speedup, ciphertext expansion), not
just the Concrete ML wrapper — recommend citing both.
```bibtex
@misc{concreteml,
  title  = {Concrete {ML}: a Privacy-Preserving Machine Learning Library using Fully Homomorphic Encryption for Data Scientists},
  author = {{Zama}},
  year   = {2022},
  note   = {\url{https://github.com/zama-ai/concrete-ml}. Evaluated and rejected: weak GPU speedup, large ciphertext expansion}
}

@misc{tfhers,
  title  = {{TFHE-rs}: A Pure Rust Implementation of the {TFHE} Scheme for Boolean and Integer Arithmetics Over Encrypted Data},
  author = {{Zama}},
  year   = {2022},
  note   = {\url{https://github.com/zama-ai/tfhe-rs}. TFHE backend underlying Concrete ML; scheme originates in Chillotti et al., J. Cryptology 33:34--91, 2020}
}
```

### thor
Non-interactive (no client boundary at all) — see scoop finding #1 below.
```bibtex
@inproceedings{thor,
  title     = {{THOR}: Secure Transformer Inference with Homomorphic Encryption},
  author    = {Moon, Jungho and Yoo, Dongwoo and Jiang, Xiaoqian and Kim, Miran},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  year      = {2025},
  doi       = {10.1145/3719027.3765150},
  note      = {Also IACR ePrint 2024/1881. Non-interactive; demonstrates full 12-layer BERT-base at 128 tokens, single GPU, ~10 minutes}
}
```

### heblas2025
Title/author order already correct; only author list was missing.
```bibtex
@article{heblas2025,
  title   = {Fast Homomorphic Linear Algebra with {BLAS}},
  author  = {Bae, Youngjin and Cheon, Jung Hee and Hanrot, Guillaume and Park, Jai Hyun and Stehl\'{e}, Damien},
  journal = {arXiv preprint arXiv:2503.16080},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.16080},
  note    = {Subsequently published in Journal of Cryptology 39 (2026), DOI 10.1007/s00145-026-09580-x}
}
```

### aegis
Bare title "AEGIS" was incomplete — full title makes scope (long-sequence, multi-GPU) explicit.
```bibtex
@article{aegis,
  title   = {{AEGIS}: Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-{GPU} Systems},
  author  = {Gong, Zhaoting and Ran, Ran and Yao, Fan and Wen, Wujie},
  journal = {arXiv preprint arXiv:2604.03425},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.03425},
  note    = {Multi-GPU HE placement with communication-computation overlap; design evidence only}
}
```

## 3. Candidate related-work literature for Section 9

None of the below has been added to `refs.bib` — presented as candidates for the author to
select from and add.

### (a) Private/secure transformer inference

**Client-assisted / interactive — top priority, matches this paper's architecture:**

- **Safhire** — "Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption"
  (Biswas, Chartier, Dhasade, Jurien, Kerriou, Kerrmarec, Lemou, Tranie, de Vos, Vujasinovic;
  arXiv 2509.01253, Sept 2025). Same decrypt/evaluate-in-clear/re-encrypt pattern as this
  paper: server evaluates linear layers under HE, client decrypts intermediates, evaluates
  nonlinearities exactly, re-encrypts. Evaluates CNNs (CIFAR-10/MNIST/ImageNet scale), not
  transformers; adds randomized output permutation to blunt client weight-reconstruction
  (a mitigation this paper doesn't claim). **Closest verified prior client-assisted/interactive
  HE work — cite and discuss explicitly.**
- **Gazelle** — "GAZELLE: A Low Latency Framework for Secure Neural Network Inference"
  (Juvekar, Vaikuntanathan, Chandrakasan; USENIX Security 2018, pp. 1651–1669). Ancestor
  pattern: HE linear layers + garbled-circuit nonlinearities, interactive. CNN-focused,
  predates transformers, but is the correct genealogical citation for "client-in-the-loop
  nonlinearity evaluation" as a pattern.

**Non-interactive HE-only (direct analog of this paper's own set-aside ablation):**

- **THOR** (entry above) — full 12-layer BERT-base, 128 tokens, single GPU, ~10 min.

**Two-party MPC / hybrid HE+MPC (field-positioning, not architecturally comparable):**

- **Iron: Private Inference on Transformers** (NeurIPS 2022)
- **BOLT** (Pang, Zhu, Möllering, Zheng, Schneider; IEEE S&P 2024)
- **BumbleBee** (NDSS 2025, demonstrates LLaMA-7B)
- **Nimbus** (NeurIPS 2024)

All four genuinely MPC-heavy (secret-sharing/OT), not client-assisted-CKKS in this paper's
sense — cite to establish that "private transformer inference" is a populated field this
paper's HE-only approach differs from, not to claim architectural equivalence.

### (b) Homomorphic encryption on genomic data

- **iDASH secure genome analysis competition 2018** (Kuo, Jiang, Tang, Wang, Bath, Bu, Wang,
  Harmanci, Zhang, Zhi, Sofia, Ohno-Machado; *BMC Medical Genomics* 13(Suppl 7):98, 2020,
  DOI 10.1186/s12920-020-0715-0). HE for GWAS; secure GWAS for 15K SNPs under 2 minutes.
- **"Private queries on encrypted genomic data"** (Çetin, Chen, Laine, Lauter, Rindal, Xia;
  *BMC Medical Genomics* 10(Suppl 2):45, 2017). HE + cuckoo hashing for encrypted
  variant/mutation queries — right citation for the beacon/variant-query angle.

Newer candidates seen (2025 multi-key-HE GWAS paper, 2026 fhEVM genomic beacon prototype):
**abstract-level only, not verified — do not cite without deeper reading.**

### (c) GPU-accelerated HE and homomorphic matrix primitives

- **FIDESlib** (already evaluated backend, entry above)
- **THOR** — diagonal-major encoding, compact packing for HE matmul
- **heblas2025** (already in `refs.bib`, correctly used)
- **Cheddar, HEonGPU** — surfaced repeatedly as FIDESlib comparison points in search results;
  **not read directly, not verified — flag as further-reading only, do not cite yet.**

## 4. Scoop / undercut risks — stated plainly

**1. THOR (CCS 2025) directly challenges the framing of this paper's non-interactive
ablation and must be addressed by name, not left implicit.** This paper's non-interactive
CKKS config completed one real-weight block, then hit a GPU memory wall at multi-block
composition. THOR — same year, comparable depth (12-layer BERT-base vs. this paper's
12 blocks), comparable prompt length (128 tokens vs. this paper's 103) — runs fully
non-interactively, on a single GPU, in ~10 minutes, with no client boundary. A reviewer who
knows THOR will ask: is the memory wall a property of non-interactive CKKS in general, or of
this project's specific library/parameter/packing choices? The evidence ledger already scopes
this correctly ("not an impossibility result for non-interactive CKKS") — but that sentence
currently names no prior work. Cite THOR there explicitly.

**2. Safhire (2025) means the client-assisted/interactive pattern itself is not novel — only
its application here is.** Frame the contribution as: applying an existing interactive
HE-inference pattern (server-linear/client-nonlinear, decrypt-evaluate-encrypt) to a
transformer, to a real released foundation model, on a genomic task, at real task-length
prompts, with an exact numeric match to a frozen plaintext oracle and a full cost
decomposition — not as inventing the pattern. This is consistent with this project's own
claim-discipline rules (conceding non-novelty where it's true strengthens the paper per
`paper-docs/AGENTS.md`). Currently unaddressed because §9 has zero citations.

**3. The Gazelle-lineage MPC field (Iron, BOLT, BumbleBee, Nimbus) is large and active** —
none architecturally comparable, but their absence from §9 reads as an oversight to a
reviewer working in this space, not a deliberate scoping choice. Cite briefly to close that
gap.

No prior work found reproduces this paper's specific combination (exact released-weight
transformer, genomic task, client-assisted CKKS, task-length prompt, measured single-block
cost decomposition). Safhire scoops the *architectural pattern*; THOR scoops the *feasibility
claim about the set-aside non-interactive alternative*. Neither is fatal — both are
addressable by citing honestly.

## Open items / could not verify

- GDPR Art. 9 / genomic re-identification literature (flagged TODO in `refs.bib` itself) —
  out of scope for this pass, still open.
- FIDESlib ISPASS 2025 poster status: corroborated via arXiv comments field and search
  context, not a primary ACM/IEEE source (both 403'd) — do one direct check before submission.
- Cheddar, HEonGPU: not verified, not cited.
- 2025 multi-key-HE GWAS paper, 2026 fhEVM beacon prototype: abstract-level only, not cited.
