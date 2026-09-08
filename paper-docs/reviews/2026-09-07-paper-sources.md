# Bibliography and citation audit — 2026-09-07

Current status: see the follow-up audit below; the initial review is retained as history.

Scope: all 22 keys cited by the current manuscript and all 22 entries in
`manuscript/source/refs.bib`. Citations were audited by key and surrounding claim rather than by
line number because the manuscript was being restructured concurrently. Records were checked
against publisher, proceedings, journal, arXiv/ePrint, software-project, and EUR-Lex primary
records. No manuscript or bibliography edits were made.

## Actionable citation-use findings

### concreteml / tfhers — misused as external support for a local comparison

The projects verify the identity and broad capabilities of Concrete ML and TFHE-rs, but they do
not establish this manuscript's comparative conclusion that GPU acceleration and ciphertext
expansion were less suitable for this graph. Cite the software when introducing the evaluated
path, then explicitly attribute the comparison to this project's evaluation (for example, “In our
backend evaluation, ...”). Do not make the repository URLs appear to be independent evidence for
that result. Also record the exact evaluated release/commit and access date in both software
entries before submission; the current year-only records are not reproducible software citations.

### iron2022 / nimbus2024 / bumblebee2025 — sentence is too distributive

All three are fairly characterized as hybrid secure two-party transformer systems, but the current
sentence grammatically attributes homomorphic encryption, secret sharing, oblivious transfer, and
specialized protocols to every cited system. The primary papers do not support that exact
all-primitives-for-all-systems claim. Use “draw on homomorphic encryption, secret sharing, and/or
oblivious-transfer-based protocols” or describe each system separately.

### heblas2025 — “shape-aware algorithms” is stronger than the source wording

The work supports a broad collection of homomorphic matrix--vector, vector--vector, and
matrix--matrix reductions implemented with BLAS. Its primary record does not use or define the
manuscript's “shape-aware algorithms” characterization. Replace that phrase with a concrete
description of the supported linear-algebra kernels.

## Bibliographic corrections

### dnabert2 — corrected

The official ICLR record uses plural **Genomes**, not singular **Genome**. Add the stable
OpenReview record. The three cited Table 4 MCC values (0.6937, 0.8677, and 0.8499) support the
manuscript's rounded 0.69, 0.87, and 0.85 comparisons on the stated benchmark splits.

```bibtex
@inproceedings{dnabert2,
  title         = {{DNABERT}-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes},
  author        = {Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana V. and Liu, Han},
  booktitle     = {International Conference on Learning Representations},
  year          = {2024},
  eprint        = {2306.15006},
  archivePrefix = {arXiv},
  url           = {https://openreview.net/forum?id=oMLQB4EZE1}
}
```

### openfhe — corrected

The current entry omits the published page range, DOI, and publisher.

```bibtex
@inproceedings{openfhe,
  title     = {{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  author    = {Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and Liu, Zeyu and Micciancio, Daniele and Quah, Ian and Polyakov, Yuriy and Saraswathy, R. V. and Rohloff, Kurt and Saylor, Jonathan and Suponitsky, Dmitriy and Triplett, Matthew and Vaikuntanathan, Vinod and Zucca, Vincent},
  booktitle = {Proceedings of the 10th Workshop on Encrypted Computing \& Applied Homomorphic Cryptography},
  pages     = {53--63},
  publisher = {Association for Computing Machinery},
  year      = {2022},
  doi       = {10.1145/3560827.3563379}
}
```

### gpu-ckks-backend — corrected

The arXiv work exists and supports GPU-resident CKKS primitives and OpenFHE interoperability, but
the current entry has neither a publication container nor a URL. It should not be represented as a
full ISPASS proceedings paper: the authors describe it as a poster there.

```bibtex
@article{gpu-ckks-backend,
  title         = {{FIDESlib}: A Fully-Fledged Open-Source {FHE} Library for Efficient {CKKS} on {GPU}s},
  author        = {Agull\'{o}-Domingo, Carlos and Vera-L\'{o}pez, \'{O}scar and Guzelhan, Seyda and Daksha, Lohit and El Jerari, Aymane and Shivdikar, Kaustubh and Agrawal, Rashmi and Kaeli, David and Joshi, Ajay and Abell\'{a}n, Jos\'{e} L.},
  journal       = {arXiv preprint arXiv:2507.04775},
  year          = {2025},
  eprint        = {2507.04775},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.04775}
}
```

### gazelle2018 — corrected

The claim about HE linear layers and garbled-circuit nonlinearities is supported. Complete the
published proceedings record.

```bibtex
@inproceedings{gazelle2018,
  title     = {{GAZELLE}: A Low Latency Framework for Secure Neural Network Inference},
  author    = {Juvekar, Chiraag and Vaikuntanathan, Vinod and Chandrakasan, Anantha},
  booktitle = {27th USENIX Security Symposium},
  pages     = {1651--1669},
  publisher = {USENIX Association},
  address   = {Baltimore, MD},
  month     = aug,
  year      = {2018},
  url       = {https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar}
}
```

### iron2022 — corrected

The current entry omits its published page range.

```bibtex
@inproceedings{iron2022,
  title     = {Iron: Private Inference on Transformers},
  author    = {Hao, Meng and Li, Hongwei and Chen, Hanxiao and Xing, Pengzhi and Xu, Guowen and Zhang, Tianwei},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {15718--15731},
  year      = {2022},
  doi       = {10.52202/068431-1143}
}
```

### nimbus2024 — corrected

The current entry omits its published page range.

```bibtex
@inproceedings{nimbus2024,
  title     = {Nimbus: Secure and Efficient Two-Party Inference for Transformers},
  author    = {Li, Zhengyi and Yang, Kang and Tan, Jin and Lu, Wen-jie and Wu, Haoqi and Wang, Xiao and Yu, Yu and Zhao, Derun and Zheng, Yancheng and Guo, Minyi and Leng, Jingwen},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {21572--21600},
  year      = {2024},
  doi       = {10.52202/079017-0680}
}
```

### bumblebee2025 — corrected

Use the full NDSS proceedings name and the publisher's author capitalization.

```bibtex
@inproceedings{bumblebee2025,
  title     = {{BumbleBee}: Secure Two-Party Inference Framework for Large Transformers},
  author    = {Lu, Wen-jie and Huang, Zhicong and Gu, Zhen and Li, Jingyu and Liu, Jian and Hong, Cheng and Ren, Kui and Wei, Tao and Chen, Wenguang},
  booktitle = {32nd Annual Network and Distributed System Security Symposium},
  publisher = {The Internet Society},
  year      = {2025},
  doi       = {10.14722/ndss.2025.230057}
}
```

### thor — corrected

The method claims are supported. The current entry omits the published page range and publisher.

```bibtex
@inproceedings{thor,
  title     = {{THOR}: Secure Transformer Inference with Homomorphic Encryption},
  author    = {Moon, Jungho and Yoo, Dongwoo and Jiang, Xiaoqian and Kim, Miran},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  pages     = {3765--3779},
  publisher = {Association for Computing Machinery},
  year      = {2025},
  doi       = {10.1145/3719027.3765150}
}
```

### gdpr — corrected

Article 9 supports the cited “special category” claim. The legal citation should carry the full
instrument title and Official Journal locator.

```bibtex
@misc{gdpr,
  title        = {Regulation ({EU}) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the Protection of Natural Persons with Regard to the Processing of Personal Data and on the Free Movement of Such Data, and Repealing Directive 95/46/{EC} (General Data Protection Regulation)},
  author       = {{European Parliament and Council of the European Union}},
  howpublished = {Official Journal of the European Union, L 119},
  pages        = {1--88},
  month        = may,
  year         = {2016},
  url          = {https://eur-lex.europa.eu/eli/reg/2016/679/oj}
}
```

### heblas2025 — corrected to the published 2026 record

This is no longer only a 2025 preprint. Cite the Journal of Cryptology version.

```bibtex
@article{heblas2025,
  title   = {Fast Homomorphic Linear Algebra with {BLAS}},
  author  = {Bae, Youngjin and Cheon, Jung Hee and Hanrot, Guillaume and Park, Jai Hyun and Stehl\'{e}, Damien},
  journal = {Journal of Cryptology},
  volume  = {39},
  number  = {3},
  pages   = {25},
  year    = {2026},
  doi     = {10.1007/s00145-026-09580-x}
}
```

### aegis — corrected to the published 2026 record

This is now an ICS 2026 proceedings paper, not only arXiv:2604.03425. The publisher record drops
the “AEGIS:” prefix from the title.

```bibtex
@inproceedings{aegis,
  title     = {Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-{GPU} Systems},
  author    = {Gong, Zhaoting and Ran, Ran and Yao, Fan and Wen, Wujie},
  booktitle = {Proceedings of the 40th ACM International Conference on Supercomputing},
  pages     = {1206--1219},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3797905.3800539}
}
```

## Verified clean

- **dnagpt2023** — verified. The architecture/task description, 0.9151 human PAS result, held-out
  6:1.5:2.5 split, and 0.62 mRNA-abundance result are supported by arXiv:2307.05628 v3.
- **deepgsr** — verified. The entry is complete and supports the historical task-specific genomic
  signal/region architecture claim; the manuscript no longer misattributes DNAGPT's 0.9151 to it.
- **xpresso** — verified. The entry is complete and supports the Xpresso dataset/model comparison;
  DNAGPT, correctly cited alongside it, is the source of the reported 0.62 result.
- **ckks2017** — verified. Entry and packed approximate-arithmetic claim are correct.
- **safhire2025** — verified. The preprint supports server-side encrypted linear evaluation,
  client-side exact nonlinear evaluation, re-encryption, and randomized shuffling.
- **idash2018** — verified. The published record and genomics-competition characterization are
  correct.
- **privategenomicqueries2017** — verified. The published record supports the HE, hashing, and
  private-set-intersection characterization.
- **gymrek2013** and **erlich2018** — verified. The records and genomic re-identification/familial
  search use are fair.

The rounded plaintext reference metrics are all source-correct: PAS accuracy 0.9151
(`dnagpt2023`); GUE MCC values 0.69/0.87/0.85 (`dnabert2`); and mRNA-abundance $r^2=0.62$
(`dnagpt2023`).

## Key reconciliation

- Citations in the manuscript with no bibliography entry: **none**.
- Bibliography entries never cited by the manuscript: **none**.

## Missing coverage

### BOLT — add to the hybrid secure-transformer paragraph

BOLT is a major peer-reviewed hybrid HE/MPC two-party transformer-inference system and is a likely
reviewer expectation alongside Iron, Nimbus, and BumbleBee. It belongs in the opening transformer
related-work paragraph.

```bibtex
@inproceedings{bolt2024,
  title     = {{BOLT}: Privacy-Preserving, Accurate and Efficient Inference for Transformers},
  author    = {Pang, Qi and Zhu, Jinhao and M\"{o}llering, Helen and Zheng, Wenting and Schneider, Thomas},
  booktitle = {2024 IEEE Symposium on Security and Privacy},
  pages     = {4753--4771},
  publisher = {IEEE},
  year      = {2024},
  doi       = {10.1109/SP54263.2024.00130}
}
```

## Follow-up audit and manuscript-owner verification

Final scope: all **24 cited keys and 24 bibliography entries**. The source specialist supplied
incremental primary-source findings; the manuscript owner completed the remaining metadata checks
and applied the corrections. Fifteen DOI-bearing records were checked against publisher-deposited
Crossref metadata (author order, title, year, venue, volume/issue and pages/article number); other
entries were checked against official proceedings, preprint, software, or legislative records.
Some publisher pages blocked automated access; those metadata checks used Crossref, with accessible
primary papers/preprints used for claim support. This is a citation audit, not an independent
replication of the cited experiments or a proof of their security guarantees.

### Corrections applied

- Corrected OpenFHE's author parsing to `{Saraswathy R.V.}`. The 2022 workshop record has 21 authors;
  later preprint author lists must not replace that proceedings author list.
- Preserved published acronym capitalization in BibTeX, including DNA, DeepGSR, mRNA and Rust.
- Removed the unsupported implication that this project benchmarked Concrete ML/TFHE-rs and found
  them less suitable. Repository evidence supports a documentation review only. Added the official
  Concrete ML inference documentation for the client/server protocol description and access dates
  for the software references. The initial review's suggestion to attribute the comparison to a
  local evaluation is superseded: no such measured comparison was established.
- Identified Safhire explicitly as a preprint and described permutation as *intended* to hinder
  reconstruction, not a demonstrated guarantee for this manuscript's protocol.
- Clarified Table IV's published comparators: DNAGPT-M and DNABERT-2, distinct from locally evaluated
  models/heads. DNAGPT Table S2 supports human AATAAA accuracy 91.51%; its human abundance section
  supports 0.62 with a 1,000-example test set. DNABERT-2's author PDF **Table 6**, not Table 4 as
  stated in the initial review above, reports 69.37/86.77/84.99 MCC points for the three cited tasks.
  The manuscript's rounded fractions remain unchanged; the additional-pretrained variant is not
  the comparator.
- Retained the verified published 2026 HE-BLAS and AEGIS metadata, and retained FIDESlib and Safhire
  as preprints. BOLT is present and cited. Earlier recommendations on hybrid-protocol wording and
  HE-BLAS kernels have been incorporated.

### Entry-by-entry disposition

All entries below are relevant to their current use; none was found to be fabricated. “Supported”
means the source supports the manuscript's bounded characterization, not that its reported results
transfer to this system.

| Key | Current claim and disposition | Primary record |
|---|---|---|
| `dnagpt2023` | Architecture, tasks and published baseline metrics supported; comparison scope clarified. | [DNAGPT v3](https://arxiv.org/html/2307.05628v3) |
| `dnabert2` | GUE benchmark and the three rounded MCC comparators supported; official ICLR title retained. | [ICLR record](https://openreview.net/forum?id=oMLQB4EZE1) |
| `deepgsr` | Prior genomic signal/region prediction work supported; not the source of DNAGPT's accuracy. | [Bioinformatics](https://doi.org/10.1093/bioinformatics/bty752) |
| `xpresso` | Sequence-based abundance prediction and dataset context supported; DNAGPT supplies its own 0.62 result. | [Cell Reports](https://doi.org/10.1016/j.celrep.2020.107663) |
| `ckks2017` | Approximate packed arithmetic and rescaling supported. | [ASIACRYPT](https://doi.org/10.1007/978-3-319-70694-8_15) |
| `openfhe` | Library capabilities supported; author formatting corrected. | [WAHC](https://doi.org/10.1145/3560827.3563379) |
| `gpu-ckks-backend` | GPU CKKS backend and OpenFHE interoperability supported; preprint status retained. | [FIDESlib](https://arxiv.org/abs/2507.04775) |
| `concreteml` | Software identity supported; no local performance comparison claimed. | [Official repository](https://github.com/zama-ai/concrete-ml) |
| `tfhers` | Boolean/integer-oriented software identity supported; no local benchmark claimed. | [Official repository](https://github.com/zama-ai/tfhe-rs) |
| `concreteml-inference` | Client attention/activations and server linear layers supported by documentation. | [Official inference documentation](https://docs.zama.org/concrete-ml/llms/inference) |
| `gazelle2018` | HE-linear/garbled-circuit nonlinear hybrid inference supported. | [USENIX](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar) |
| `iron2022` | Private transformer inference and specialized linear/nonlinear protocols supported. | [NeurIPS](https://proceedings.neurips.cc/paper/2022/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract.html) |
| `bolt2024` | Hybrid secure transformer inference supported; conference details verified. | [IEEE S&P](https://doi.org/10.1109/SP54263.2024.00130) |
| `nimbus2024` | Two-party transformer inference and nonlinear approximations supported. | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/264a9b3ce46abdf572dcfe0401141989-Abstract-Conference.html) |
| `bumblebee2025` | Two-party transformer inference and optimized matrix/nonlinear protocols supported. | [NDSS](https://www.ndss-symposium.org/ndss-paper/bumblebee-secure-two-party-inference-framework-for-large-transformers/) |
| `safhire2025` | Client-decrypted nonlinearities and randomized permutation supported; security wording narrowed. Unusual author spellings match the primary record. | [Safhire preprint](https://arxiv.org/abs/2509.01253) |
| `thor` | Non-interactive HE transformer architecture and compact diagonal-major packing supported. | [ACM CCS](https://doi.org/10.1145/3719027.3765150) |
| `idash2018` | Secure-genomics competition characterization supported; publication year is 2020 despite the 2018 competition. | [BMC Medical Genomics](https://doi.org/10.1186/s12920-020-0715-0) |
| `privategenomicqueries2017` | HE, hashing and private-set-intersection genomic querying supported. | [BMC Medical Genomics](https://doi.org/10.1186/s12920-017-0276-z) |
| `gdpr` | Genetic-data special-category characterization supported by Article 9; no compliance certification claimed. | [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj) |
| `gymrek2013` | Surname-based genomic re-identification motivation supported. | [Science](https://doi.org/10.1126/science.1229566) |
| `erlich2018` | Long-range familial-search privacy motivation supported. | [Science](https://doi.org/10.1126/science.aau4832) |
| `heblas2025` | BLAS-based homomorphic linear-algebra kernels supported; final journal year is 2026, notwithstanding the internal key. | [Journal of Cryptology](https://doi.org/10.1007/s00145-026-09580-x) |
| `aegis` | Multi-GPU placement and communication/computation overlap supported; final ICS pages 1206–1219 verified. | [ACM ICS](https://doi.org/10.1145/3797905.3800539), [author preprint](https://arxiv.org/abs/2604.03425) |

### Final reconciliation and render

- 24 cited keys, 24 bibliography entries, 24 rendered references; no missing or uncited entries.
- Build and all four manuscript linters pass. PDF remains 12 pages with 9 figures, 6 tables and
  4 algorithms. Figure 2 now enforces at least 3 pt label-to-box clearance in the figure generator.
- Inspected the complete page contact sheet and full-size Figure 2/bibliography pages. No blank
  body columns on pages 6–8, missing figures, clipped Figure 2 labels, or reference overflow found.
- No overfull-box or undefined-reference warnings. Some underfull-box spacing warnings remain;
  these are not missing content or compilation failures.
