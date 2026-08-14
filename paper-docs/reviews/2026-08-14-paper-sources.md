# Paper sources review — 2026-08-14

## Academic verdict

The paper has a defensible contribution and a substantially stronger related-work posture than the
earlier drafts. It correctly declines to claim the client-assisted pattern as novel, concedes that
THOR is faster, and frames the contribution as an auditable genomic-model case study with an
independent numerical oracle. It is close to a solid paper, not a perfect one.

Before submission, four source-facing issues should be fixed because reviewers can reasonably treat
them as load-bearing:

1. The THOR comparison is useful but is not a controlled comparison. The introduction's claim that
   THOR uses "this study's cryptographic parameters" and "this study's accelerator" is too strong.
   THOR reports the same ring degree (`2^16`), 13 available levels, nominal 128-bit security, one
   input, a 12-layer/768-wide model, 128 tokens, and a single NVIDIA A100. It does **not** establish
   the same A100 SKU/memory capacity, modulus chain, scale, library, CPU path, model semantics, or
   nonlinear circuit. THOR uses Liberate.FHE, a BERT-base encoder, polynomial nonlinearities, and
   bootstrapping; this work uses FIDESlib/OpenFHE, a causal DNAGPT decoder, exact client
   nonlinearities, and no bootstrapping. The `602.26 s` versus `6683 s` comparison is fair as a
   cross-system contextual comparison and supports "roughly an order of magnitude slower," but it
   should be qualified at its first occurrence as reported, uncontrolled, and cross-library.
2. The model-extraction statement is stronger than the evidence. "Weight recovery becomes
   inexpensive per-segment regression" is uncited and unmeasured. Some MLP affine maps do become
   ordinary linear-system identification if the client observes sufficiently diverse, full-rank
   input/output pairs. That does not prove inexpensive recovery of all weights: attention reveals
   composite relations and has factorization symmetries, query activations may be rank deficient,
   and no extraction experiment or sample-complexity analysis is reported. The defensible claim is
   that the transcript creates a plausible chosen-query extraction surface and makes at least some
   affine segments directly identifiable; therefore model confidentiality is not claimed.
3. The GSR comparison is disclosed but still over-read in the surrounding prose. The released
   classifier is evaluated on all `22,604` examples, whereas DNAGPT's `0.9151` is the DNAGPT-M
   accuracy on a held-out 25% split (Table S2; train:validation:test `6:1.5:2.5`). The all-example run
   includes examples used to train the released head, so `0.9124` is pipeline/model-fidelity evidence,
   not an independent held-out generalization estimate. The sentence saying every task uses its
   canonical test split is false for GSR and uncertain for the reconstructed mRNA split. The ledger
   also calls `22,604` a "canonical test split size," contradicting its own reference-scope note.
4. Two bibliography records are materially wrong: `moai2025` has no authors and now has an ICLR
   2026 publication record, while `dnaembeddinginversion2026` gives the wrong first names (the
   primary record is Sofiane Ouaari and Jules Kreuer, not Mehdi Ouaari and Seán Kreuer). The empty
   MOAI author already produces a BibTeX sorting warning in `main.blg`.

## Plaintext reference metrics

- **GSR:** `0.9151` is verified in DNAGPT Table S2 for DNAGPT-M, Human PAS(AATAAA). The manuscript's
  held-out-quarter versus all-example caveat is accurate. Treat the local result only as release and
  pipeline fidelity because the evaluation is not held out from the released head's training.
- **GUE:** the exact DNABERT-2 rows are `69.37`, `86.77`, and `84.99` MCC for core-promoter-all,
  promoter-all, and reconstructed splice sites. Rounding these to `.69`, `.87`, and `.85` is fine.
  The current prose is attribution-fair because it identifies these as DNABERT-2's own rows and notes
  that other models score higher on two of the three. The locally trained head and three-of-28-task
  scope remain necessary qualifiers.
- **mRNA:** DNAGPT explicitly says the human `r^2=0.62` result is from DNAGPT-M and improves the
  Xpresso `0.59` baseline. This work evaluates the released head fine-tuned from DNAGPT-H and a
  reconstructed last-1,000 split whose gene identity is not verified. The table footnote now states
  the backbone mismatch, so the number is acceptable as model-fidelity context only, not a
  reproduction of the published result.
- **Ledger consistency:** change the interpretation, not the evidence: `base.gsr_n=22604` is the full
  balanced corpus used locally, not a canonical test split. The general statement that all tasks use
  canonical test splits should be replaced by task-specific split language.

Primary records checked: DNAGPT [arXiv 2307.05628](https://arxiv.org/abs/2307.05628), DNABERT-2
[OpenReview](https://openreview.net/forum?id=oMLQB4EZE1), and Xpresso
[Cell Reports](https://doi.org/10.1016/j.celrep.2020.107663).

## Claim-to-source and attribution findings

- **THOR (`602.26 s`): verified, with a fairness correction.** THOR Table 6 totals `602.26 s` for one
  128-token BERT-base input on one A100. Section 7.1 reports ring degree `2^16`, `2^15` slots,
  128-bit security, and 13 levels before bootstrapping. It does not establish a hardware-identical or
  parameter-identical setup beyond those named dimensions. Cite it as an external reported number,
  not a matched benchmark. Primary: [IACR ePrint 2024/1881](https://eprint.iacr.org/2024/1881).
- **MOAI (`52.8%`): verified, but do not mix its experiments.** MOAI Table 4 reports `283.95 s`
  versus THOR's `602.26 s` on a single A100 and calls this a `52.8%` reduction. Its Table 3 `141.3 s`
  (`2.36 min`) result is a different, 256-input-amortized H200 experiment. The manuscript currently
  quotes only the 52.8% comparison, which is supportable; retain the authors' attribution and avoid
  implying that the H200 amortized figure is a single-query A100 result. Primary:
  [IACR ePrint 2025/991](https://eprint.iacr.org/2025/991).
- **NEXUS batching:** the statement that its headline number is amortized while single-prediction
  latency is much higher is credible, but `\cite{nexus2025}` alone does not make the derivation easy
  to audit. Give the batch size/table or also cite THOR's explicit single-request calculation.
- **Concrete ML / TFHE-rs:** the repositories verify that the software exists and that TFHE-rs is a
  Boolean/integer-oriented TFHE implementation. They do not support this project's comparative
  claim that GPU availability and ciphertext expansion were less suitable, nor do they by themselves
  establish the exact evaluated transformer boundary protocol. Attribute those as project-specific
  engineering observations and cite the precise Concrete ML hybrid-model documentation if that API
  behavior matters.
- **FIDESlib serialization:** the FIDESlib paper supports GPU CKKS primitives and OpenFHE
  interoperability. It does not support the project-specific observation that ciphertext
  serialization was unavailable. Keep that sentence explicitly framed as an observed API constraint,
  backed by project evidence rather than by the FIDESlib citation.
- **CKKS decryption security:** the limitations discuss adversarially selected boundary ciphertexts
  and absent noise flooding but cite no cryptographic source. Add Li and Micciancio, "On the Security
  of Homomorphic Encryption on Approximate Numbers," EUROCRYPT 2021, and state clearly that the
  evaluated semi-honest protocol excludes the active attack. Numerical error headroom is not a
  security proof for a flooding parameter.
- **Novelty fairness:** the current paper is fair in saying Safhire precedes the server-linear /
  client-nonlinear pattern and THOR/MOAI outperform it on latency. Keep the novelty claim at the
  application-and-evidence level: released genomic transformer, task-defined 103-token prompt,
  exact released nonlinearities, full-model execution, and independently checked oracle. Do not
  imply novelty of client assistance, CKKS transformer inference, or encrypted matrix layouts.

## Bibliography audit

Each record below was checked against a primary publisher, proceedings, repository, legal record, or
author preprint. The BibTeX shown is the recommended record, not an edit made by this review.

### `dnagpt2023` — verified

The record exists and supports the model architecture, task families, GSR metric, and mRNA metric.

```bibtex
@article{dnagpt2023,
  title={DNAGPT: A Generalized Pre-trained Tool for Versatile DNA Sequence Analysis Tasks},
  author={Zhang, Daoan and Zhang, Weitong and Zhao, Yu and Zhang, Jianguo and He, Bing and Qin, Chenchen and Yao, Jianhua},
  journal={arXiv preprint arXiv:2307.05628}, year={2023},
  url={https://arxiv.org/abs/2307.05628}
}
```

### `dnabert2` — corrected

Because the entry cites the ICLR version, use the conference title's plural **Genomes**, the formal
conference name, and the OpenReview URL.

```bibtex
@inproceedings{dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana V. and Liu, Han},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}, url={https://openreview.net/forum?id=oMLQB4EZE1}
}
```

### `deepgsr` — verified

```bibtex
@article{deepgsr,
  title={DeepGSR: an optimized deep-learning structure for the recognition of genomic signals and regions},
  author={Kalkatawi, Manal and Magana-Mora, Arturo and Jankovic, Boris and Bajic, Vladimir B.},
  journal={Bioinformatics}, volume={35}, number={7}, pages={1125--1132}, year={2019},
  doi={10.1093/bioinformatics/bty752}
}
```

### `xpresso` — verified

```bibtex
@article{xpresso,
  title={Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks},
  author={Agarwal, Vikram and Shendure, Jay}, journal={Cell Reports}, volume={31}, number={7},
  pages={107663}, year={2020}, doi={10.1016/j.celrep.2020.107663}
}
```

### `ckks2017` — verified

```bibtex
@inproceedings{ckks2017,
  title={Homomorphic Encryption for Arithmetic of Approximate Numbers},
  author={Cheon, Jung Hee and Kim, Andrey and Kim, Miran and Song, Yongsoo},
  booktitle={Advances in Cryptology -- ASIACRYPT 2017}, series={Lecture Notes in Computer Science},
  volume={10624}, pages={409--437}, year={2017}, doi={10.1007/978-3-319-70694-8_15}
}
```

### `openfhe` — corrected

The current conference record omits pages and DOI. Its author list correctly matches the 2022
conference version rather than the expanded 2024 preprint.

```bibtex
@inproceedings{openfhe,
  title={{OpenFHE}: Open-Source Fully Homomorphic Encryption Library},
  author={Al Badawi, Ahmad and Bates, Jack and Bergamaschi, Flavio and Cousins, David Bruce and Erabelli, Saroja and Genise, Nicholas and Halevi, Shai and Hunt, Hamish and Kim, Andrey and Lee, Yongwoo and Liu, Zeyu and Micciancio, Daniele and Quah, Ian and Polyakov, Yuriy and Saraswathy, R. V. and Rohloff, Kurt and Saylor, Jonathan and Suponitsky, Dmitriy and Triplett, Matthew and Vaikuntanathan, Vinod and Zucca, Vincent},
  booktitle={Proceedings of the 10th Workshop on Encrypted Computing \& Applied Homomorphic Cryptography},
  pages={53--63}, year={2022}, doi={10.1145/3560827.3563379}
}
```

### `gpu-ckks-backend` — corrected

The ISPASS proceedings record is now available; the arXiv-only/poster-note record is stale.

```bibtex
@inproceedings{gpu-ckks-backend,
  title={{FIDESlib}: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs},
  author={Agull\'{o}-Domingo, Carlos and Vera-L\'{o}pez, \'{O}scar and Guzelhan, Seyda and Daksha, Lohit and El Jerari, Aymane and Shivdikar, Kaustubh and Agrawal, Rashmi and Kaeli, David and Joshi, Ajay and Abell\'{a}n, Jos\'{e} L.},
  booktitle={2025 IEEE International Symposium on Performance Analysis of Systems and Software},
  pages={365--367}, year={2025}, doi={10.1109/ISPASS64960.2025.00045},
  url={https://arxiv.org/abs/2507.04775}
}
```

### `concreteml` — misused

The repository metadata is real. The citation does not substantiate the paper's project-specific
performance and suitability conclusion.

```bibtex
@misc{concreteml,
  title={{Concrete ML}: Privacy-Preserving Machine Learning with Fully Homomorphic Encryption},
  author={{Zama}}, year={2022}, howpublished={Software repository},
  url={https://github.com/zama-ai/concrete-ml}
}
```

### `tfhers` — misused

The repository supports the Boolean/integer TFHE description, not the comparative selection result.

```bibtex
@misc{tfhers,
  title={{TFHE-rs}: A Pure Rust Implementation of the {TFHE} Scheme for Boolean and Integer Arithmetic},
  author={{Zama}}, year={2022}, howpublished={Software repository},
  url={https://github.com/zama-ai/tfhe-rs}
}
```

### `gazelle2018` — verified

```bibtex
@inproceedings{gazelle2018,
  title={{GAZELLE}: A Low Latency Framework for Secure Neural Network Inference},
  author={Juvekar, Chiraag and Vaikuntanathan, Vinod and Chandrakasan, Anantha},
  booktitle={27th USENIX Security Symposium (USENIX Security 18)}, pages={1651--1669},
  publisher={USENIX Association}, year={2018},
  url={https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar}
}
```

### `iron2022` — verified

```bibtex
@inproceedings{iron2022,
  title={Iron: Private Inference on Transformers},
  author={Hao, Meng and Li, Hongwei and Chen, Hanxiao and Xing, Pengzhi and Xu, Guowen and Zhang, Tianwei},
  booktitle={Advances in Neural Information Processing Systems}, volume={35}, pages={15718--15731},
  year={2022}, doi={10.52202/068431-1143}
}
```

### `nimbus2024` — verified

```bibtex
@inproceedings{nimbus2024,
  title={Nimbus: Secure and Efficient Two-Party Inference for Transformers},
  author={Li, Zhengyi and Yang, Kang and Tan, Jin and Lu, Wen-jie and Wu, Haoqi and Wang, Xiao and Yu, Yu and Zhao, Derun and Zheng, Yancheng and Guo, Minyi and Leng, Jingwen},
  booktitle={Advances in Neural Information Processing Systems}, volume={37}, pages={21572--21600},
  year={2024}, doi={10.52202/079017-0680}
}
```

### `bumblebee2025` — verified

```bibtex
@inproceedings{bumblebee2025,
  title={BumbleBee: Secure Two-party Inference Framework for Large Transformers},
  author={Lu, Wen-jie and Huang, Zhicong and Gu, Zhen and Li, Jingyu and Liu, Jian and Hong, Cheng and Ren, Kui and Wei, Tao and Chen, WenGuang},
  booktitle={Network and Distributed System Security Symposium}, year={2025},
  doi={10.14722/ndss.2025.230057}
}
```

### `safhire2025` — verified

The preprint supports the closest-antecedent description, including server-side homomorphic linear
layers, client-side nonlinear evaluation, and randomized shuffling.

```bibtex
@article{safhire2025,
  title={Practical and Private Hybrid ML Inference with Fully Homomorphic Encryption},
  author={Biswas, Sayan and Chartier, Philippe and Dhasade, Akash and Jurien, Tom and Kerriou, David and Kerrmarec, Anne-Marie and Lemou, Mohammed and Tranie, Franklin and de Vos, Martijn and Vujasinovic, Milos},
  journal={arXiv preprint arXiv:2509.01253}, year={2025},
  url={https://arxiv.org/abs/2509.01253}
}
```

### `thor` — corrected

The runtime and parameter claims are supported. Add the published page range.

```bibtex
@inproceedings{thor,
  title={{THOR}: Secure Transformer Inference with Homomorphic Encryption},
  author={Moon, Jungho and Yoo, Dongwoo and Jiang, Xiaoqian and Kim, Miran},
  booktitle={Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  pages={3765--3779}, year={2025}, doi={10.1145/3719027.3765150},
  url={https://eprint.iacr.org/2024/1881}
}
```

### `idash2018` — verified

```bibtex
@article{idash2018,
  title={{iDASH} secure genome analysis competition 2018: blockchain genomic data access logging, homomorphic encryption on {GWAS}, and DNA segment searching},
  author={Kuo, Tsung-Ting and Jiang, Xiaoqian and Tang, Haixu and Wang, XiaoFeng and Bath, Tyler and Bu, Diyue and Wang, Lei and Harmanci, Arif and Zhang, Shaojie and Zhi, Degui and Sofia, Heidi J. and Ohno-Machado, Lucila},
  journal={BMC Medical Genomics}, volume={13}, number={Suppl 7}, pages={98}, year={2020},
  doi={10.1186/s12920-020-0715-0}
}
```

### `privategenomicqueries2017` — verified

```bibtex
@article{privategenomicqueries2017,
  title={Private queries on encrypted genomic data},
  author={\c{C}etin, Gizem S. and Chen, Hao and Laine, Kim and Lauter, Kristin and Rindal, Peter and Xia, Yuhou},
  journal={BMC Medical Genomics}, volume={10}, number={Suppl 2}, pages={45}, year={2017},
  doi={10.1186/s12920-017-0276-z}
}
```

### `gdpr` — verified

The legal source supports the special-category statement for genetic data.

```bibtex
@misc{gdpr,
  title={Regulation ({EU}) 2016/679 of the European Parliament and of the Council},
  author={{European Parliament and Council of the European Union}},
  howpublished={Official Journal of the European Union}, year={2016},
  url={https://eur-lex.europa.eu/eli/reg/2016/679/oj}
}
```

### `gymrek2013` — verified

```bibtex
@article{gymrek2013,
  title={Identifying Personal Genomes by Surname Inference},
  author={Gymrek, Melissa and McGuire, Amy L. and Golan, David and Halperin, Eran and Erlich, Yaniv},
  journal={Science}, volume={339}, number={6117}, pages={321--324}, year={2013},
  doi={10.1126/science.1229566}
}
```

### `erlich2018` — verified

```bibtex
@article{erlich2018,
  title={Identity inference of genomic data using long-range familial searches},
  author={Erlich, Yaniv and Shor, Tal and Pe'er, Itsik and Carmi, Shai},
  journal={Science}, volume={362}, number={6415}, pages={690--694}, year={2018},
  doi={10.1126/science.aau4832}
}
```

### `heblas2025` — corrected

The published record is volume 39, article 25; include the article number.

```bibtex
@article{heblas2025,
  title={Fast Homomorphic Linear Algebra with {BLAS}},
  author={Bae, Youngjin and Cheon, Jung Hee and Hanrot, Guillaume and Park, Jai Hyun and Stehl\'{e}, Damien},
  journal={Journal of Cryptology}, volume={39}, pages={25}, year={2026},
  doi={10.1007/s00145-026-09580-x}
}
```

### `aegis` — corrected

The work exists and says it was accepted at ICS 2026, but an official proceedings DOI/page record
was not found. Until one is available, cite the primary arXiv record rather than presenting an
incomplete proceedings record.

```bibtex
@misc{aegis,
  title={{AEGIS}: Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-{GPU} Systems},
  author={Gong, Zhaoting and Ran, Ran and Yao, Fan and Wen, Wujie}, year={2026},
  eprint={2604.03425}, archivePrefix={arXiv}, note={Accepted at ICS 2026},
  url={https://arxiv.org/abs/2604.03425}
}
```

### `nexus2025` — verified

```bibtex
@inproceedings{nexus2025,
  title={Secure Transformer Inference Made Non-interactive},
  author={Zhang, Jiawen and Yang, Xinpeng and He, Lipeng and Chen, Kejia and Lu, Wen-jie and Wang, Yinghao and Hou, Xiaoyang and Liu, Jian and Ren, Kui and Yang, Xiaohu},
  booktitle={Network and Distributed System Security Symposium}, year={2025},
  doi={10.14722/ndss.2025.230868}
}
```

### `bolt2024` — corrected

The current record omits the IEEE DOI.

```bibtex
@inproceedings{bolt2024,
  title={{BOLT}: Privacy-Preserving, Accurate and Efficient Inference for Transformers},
  author={Pang, Qi and Zhu, Jinhao and M\"{o}llering, Helen and Zheng, Wenting and Schneider, Thomas},
  booktitle={2024 IEEE Symposium on Security and Privacy}, pages={4753--4771}, year={2024},
  doi={10.1109/SP54263.2024.00130}
}
```

### `moai2025` — corrected

The current entry has no author and triggers a BibTeX warning. The primary record now identifies all
authors and publication at ICLR 2026.

```bibtex
@inproceedings{moai2025,
  title={{MOAI}: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference},
  author={Zhang, Linru and Wang, Xiangning and Sim, Jun Jie and Huang, Zhicong and Zhong, Jiahao and Wang, Huaxiong and Duan, Pu and Lam, Kwok Yan},
  booktitle={The Fourteenth International Conference on Learning Representations}, year={2026},
  url={https://openreview.net/forum?id=qJn4HtTzhH}
}
```

### `dnaembeddinginversion2026` — corrected

The current author first names are wrong. The arXiv primary record lists Sofiane Ouaari and Jules
Kreuer. The source does support near-perfect reconstruction from per-token DNA-model embeddings.

```bibtex
@misc{dnaembeddinginversion2026,
  title={How Private Are {DNA} Embeddings? Inverting Foundation Model Representations of Genomic Sequences},
  author={Ouaari, Sofiane and Kreuer, Jules and Pfeifer, Nico}, year={2026},
  eprint={2603.06950}, archivePrefix={arXiv},
  url={https://arxiv.org/abs/2603.06950}
}
```

## Citation inventory

- Citations used in the manuscript but missing from `refs.bib`: **none**.
- Entries in `refs.bib` never cited in the manuscript: **none**.
- Mechanical sweep: **26 cited keys, 26 bibliography keys**, with exact set equality.
- Build warning: `moai2025` lacks both `author` and `key`, so BibTeX reports
  `Warning--to sort, need author or key in moai2025`.

## Missing coverage worth adding

These are not a demand for encyclopedic related work. The first two materially improve novelty
positioning; the last two close specific technical gaps.

1. **Powerformer** — Dongjin Park, Eunsang Lee, and Joon-Woo Lee, ACL 2025,
   [primary record](https://aclanthology.org/2025.acl-long.543/). It replaces softmax and LayerNorm
   with distilled power/linear functions, approximates GELU/tanh, and reports a 45% encrypted-time
   reduction. Add it beside THOR and MOAI to distinguish this paper's exact released nonlinearities
   from retraining/model modification. This is the highest-priority omission because MOAI itself
   compares against Powerformer.
2. **THE-X** — Tianyu Chen et al., Findings of ACL 2022,
   [primary record](https://aclanthology.org/2022.findings-acl.277/). It is an early encrypted
   transformer workflow and offloads ReLU to the client. Add one historical sentence before the
   modern non-interactive systems; it helps avoid making Safhire appear to be the only interactive
   predecessor.
3. **ELLMo** — Seyda Nur Guzelhan et al., IACR ePrint 2026/198,
   [primary record](https://eprint.iacr.org/2026/198). It is a timely packing- and depth-aware
   encrypted transformer system from the same broader GPU-CKKS ecosystem, with fused packing,
   Statistical-max Softmax, DelayNorm, up to 46% fewer bootstraps, and a 1.4x BERT-Tiny speedup at
   0--1.5% accuracy loss. Mention it in the matrix/packing optimization paragraph if the venue's
   currency expectations extend through mid-2026.
4. **Li and Micciancio** — Baiyu Li and Daniele Micciancio, EUROCRYPT 2021,
   [primary preprint](https://eprint.iacr.org/2020/1533), DOI
   `10.1007/978-3-030-77870-5_23`. It provides the necessary cryptographic basis for the discussion
   of CKKS decryption exposure and countermeasures. Cite it in the limitations, while making clear
   that the evaluated semi-honest server is not allowed to mount the active chosen-ciphertext
   behavior under discussion.

## Bottom line

The paper's scholarly value does not depend on winning the speed comparison or inventing the
protocol pattern. Its credible contribution is the unusually explicit execution-and-verification
account for a released genomic transformer. Fix the false-equivalence wording around THOR, narrow
the unmeasured weight-recovery claim, label the all-example GSR number as fidelity rather than
held-out performance, repair the bibliography metadata, and add Powerformer/THE-X. Those changes
are enough for a solid, honest academic positioning without turning the paper into a survey.
