Comments from collegue we need to address:

Format: "Section or specific text from the paper" > comment
---

"Abstract" > the abstract is not easy to follow, needs to ease the reader a bit

"We validate DNAGPT on three genomic task families and construct an independently checked numerical reference." > would rephrase, I understand what is meant by validate here but its a bit confusing

"Key Points" > ειναι πολυ technically loaded, πρεπει να γινουν πιο easy to follow

"The released weights and nonlinear formulas are preserved unmodified, and the one evaluated prompt reproduces the reference label with 8.5610−9 relative margin error 6.67 orders of magnitude below the 410−2 tolerance." > Add: “6.67 orders of magnitude below the 410−2 tolerance”

"The validation chain separately records released float32 behavior, checks an independent NumPy float64 reference against a float64-cast PyTorch shadow, and compares encrypted execution with that reference." > very technical, a reader won't be able to understand why that matters.

"One in-process execution takes 1.86 hours and requires an online key holder, so the evaluated form is not interactive" > The protocol is interactive in the cryptographic/protocol sense—it repeatedly communicates with the key holder. What you mean is that it is not fast enough for interactive use.

"The data owner tokenizes and embeds a genomic fragment locally, encrypts the embedded vectors, and retains the secret key. The provider evaluates model linear algebra on ciphertexts; the data owner evaluates declared nonlinear functions on CPU. The measured implementation emulates both roles in one process. The result panel reports one measured execution at the 103-token genomic-signal prompt." > The provider does more than linear algebra: it also computes encrypted LayerNorm statistics, query–key products, attention-weight–value products, residuals, etc. Some of these involve ciphertext–ciphertext multiplication and are not simply “linear algebra.” A safer phrase is “the provider evaluates the HE-compatible model operations on ciphertexts” or “the provider evaluates the encrypted arithmetic between client boundaries.”

"The same sequence is the input on which genomic foundation models derive their value: DNAGPT serves classification, regression, and generation tasks through a common transformer backbone" > DNAGPT doesnt use human data so we need to say it a bit more carefully

" When a genomic model is available only as a service, conventional inference requires the data owner to send the sequence, or a representation of it, to the provider.When that model is served rather than distributed, a data owner without accelerator hardware must otherwise disclose the sequence to the provider or forgo the inference. Sending plaintext embeddings does not necessarily solve the privacy problem: recent work has shown that per-token representations from DNA foundation models can permit near-perfect reconstruction of the underlying sequence. Nor is handing over an embedding a way out: per-token embeddings from DNA foundation models have been shown to permit near-perfect reconstruction of the underlying sequence" > Replace: “Nor is handing over an embedding a way out: per-token embeddings from DNA foundation models have bee…” with “Sending plaintext embeddings does not necessarily solve the privacy problem: recent work has shown t…”

"DNAGPT is public and can be run locally; we use it because its release makes the experiment reproducible and auditableDNAGPT’s public release makes the experiment auditable and also makes this particular artifact available for local execution." > "Replace: “DNAGPT’s public release makes the experiment auditable and also makes this particular artifact avail…” with “DNAGPT is public and can be run locally; we use it because its release makes the experiment reproduc…”
Ilias Georgakopoulos-Soares
Ilias Georgakopoulos-Soares
1:38 AM Aug 17
if we could make the reasoning even stronger.
"

"We therefore ask whether a complete released genomic transformer can be executed under homomorphic encryption, and what limits its numerical fidelity, memory use, and runtime. We separate two questions. Feasibility asks whether the complete computation can execute within the cryptographic and memory budget while meeting the predefined numerical criterion for the evaluated case. Practicality asks whether the resulting resource cost is suitable for the intended use. A numerically agreeing but slow execution can therefore establish complete-model feasibility for the tested case without establishing a practical service.Against this deployment premise, we ask: can inference with a released genomic foundation model be executed under homomorphic encryption, and where would correctness, performance, or memory prevent a complete encrypted deployment? The question contains two verdicts that should not be collapsed. Feasibility asks whether the released computation can be executed faithfully within the available cryptographic and memory budget. Practicality asks whether its resource cost supports a target use. A correct but slow encrypted path is therefore evidence: it resolves the first question while locating the work needed to change the second." > Replace: “Against this deployment premise, we ask: can inference with a released genomic foundation model be e…” with “We therefore ask whether a complete released genomic transformer can be executed under homomorphic e…”

"The main cryptographic difficulty arises from transformer operations that are not directly supported by low-depth CKKS arithmetic. CKKS efficiently supports the packed additions, multiplications, and rotations needed for dense linear transformationsThe technical difficulty is concentrated where a transformer departs from affine arithmetic. CKKS supports approximate packed arithmetic and maps naturally to dense projections, rotations, and accumulations" > Replace: “The technical difficulty is concentrated where a transformer departs from affine arithmetic. CKKS su…” with “The main cryptographic difficulty arises from transformer operations that are not directly supported…”

"Numerical agreementCorrectness holds:" > Replace: “Correctness” with “Numerical agreement”

Accelerator memory holds as well, and for a structural reason: because multiplicative depth is bounded by the longest segment between client boundaries instead of by network depth, it stays at 13 for the complete model, device-wide GPU memory used reaches 9839 MiB without growth across the twelve blocks, and no homomorphic bootstrap is required at any point. > rephrase to make it easier to understand.

"Complete-model graph feasibility." > Add: “graph”

"Independently checked numerical evaluation. Model-faithful, reproducible evaluation." > Replace: “Model-faithful, reproducible evaluation.” with “Independently checked numerical evaluation.”

"The dense maps use fixed model weights, whereas LayerNorm, attention, and GELU introduce data-dependent products or nonlinear functions." > check if needs fixing,also calling attention a source of nonlinear functions is imprecise

"The release includes fine-tuned heads for genomic-signal recognition and mRNA abundance regression. It does not include a head for the genome-understanding benchmark, so Section 4 evaluates that backbone with a locally fine-tuned linear head. Each of these families predates the model: signal recognition by task-specific convolutional architectures (Kalkatawi et al. 2019), abundance regression likewise (Agarwal and Shendure 2020), and the understanding benchmark by the encoder-based model that introduced it (Zhou et al. 2024). They first establish model fidelity; the encrypted study then uses the signal-recognition path as its task-length anchor." > do we need this?
