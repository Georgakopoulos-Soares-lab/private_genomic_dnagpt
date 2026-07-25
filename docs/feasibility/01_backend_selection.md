# Backend selection

Scheme A's requirement was stricter than ordinary private inference: embedded input
encrypted, block output encrypted, and no intermediate decryption anywhere in the
arithmetic pass. DNAGPT's block contains LayerNorm, causal softmax attention, GELU,
linear maps, and residual additions (`DNAGPT/dna_gpt/model/gpt.py`). Scheme B (the
active path since 2026-07-25, see
[05_architecture_options.md](05_architecture_options.md)) relaxes only the
zero-intermediate-decrypt clause, and only for the data-owning client at pre-declared
nonlinearity boundaries; the backend selection below is unchanged for both schemes.

## Decision

| Evaluated path | Verdict | Reason |
|---|---|---|
| Concrete ML hybrid LLM | Rejected as a backend | Its documented LLM protocol runs linear layers on the FHE server and nonlinear layers on the client, same *shape* now adopted as Scheme B — but its TFHE-rs backend only reports 1.2-2x GPU speedup and 2.2-18MB/token ciphertext expansion, far weaker than FIDESlib CKKS's proven GPU numbers. Scheme B keeps FIDESlib/OpenFHE CKKS end to end and adds the same client-decrypt-boundary shape without changing backend. |
| TenSEAL CKKS | Not selected | Its public API does not expose the bootstrap and nonlinear-function path needed for repeated transformer blocks. |
| OpenFHE CKKS | Selected correctness backend | It exposes encrypted rotations, Chebyshev function evaluation, division, and approximate CKKS bootstrapping in one context. The complete local block passes. |
| FIDESlib CKKS | Selected performance backend | Current FIDESlib supplies CUDA CKKS operations, hoisted rotation, bootstrap, OpenFHE interop, and optional multi-GPU execution. |

This is a selection among the evaluated software paths, not a claim that CKKS is the
only FHE scheme capable of expressing the graph.

## Concrete ML

Concrete ML's [LLM inference documentation](https://docs.zama.org/concrete-ml/llms/inference)
states that the client executes nonlinear layers, including attention and activations,
while the server executes linear layers. It reports about 300 seconds per GPT-2 token on
CPU and about 11 seconds on GPU for that hybrid path. `[V]`

That protocol can protect data from the server, but it did not satisfy Scheme A's
explicit zero-intermediate-decryption condition. The rejection applies to the Concrete
ML/TFHE-rs *backend* — weak measured GPU speedup, Boolean/integer-only arithmetic, large
ciphertext expansion — not to the client-decrypt-boundary *protocol shape*, which
Scheme B now adopts on top of FIDESlib/OpenFHE CKKS instead. Concrete ML can implement
nonlinear functions as programmable-bootstrapped table lookups in other circuits.

## TenSEAL

[TenSEAL's public project](https://github.com/OpenMined/TenSEAL) provides encrypted
vector/tensor arithmetic but does not document a bootstrap or general encrypted
Chebyshev-function API. DNAGPT requires repeated inverse-square-root, exponential,
reciprocal, and GELU approximations across 12 blocks. OpenFHE already demonstrates those
operations and refresh in this repository, so a second CPU wrapper adds no new claim.

## OpenFHE CKKS

OpenFHE's official project describes CKKS support for approximate real arithmetic and
[approximate bootstrapping](https://github.com/openfheorg/openfhe-development). Its
official Python examples include
[iterative CKKS bootstrapping](https://github.com/openfheorg/openfhe-python/blob/main/examples/pke/iterative-ckks-bootstrapping.py).

Local measurements in this repository establish:

- `[V]` BSGS linear, LayerNorm, softmax, and GELU primitives pass the `4e-2` gate at
  `HEStd_128_classic`; see `results/runs/fhe_operator_matrix_d8_20260724.json`.
- `[V]` A depleted ciphertext is refreshed and supports nine more multiplicative levels;
  see `results/runs/fhe_bootstrap_d8_20260724.json`.
- `[V]` The embedded-input-to-block-output `D=8`, `T=4`, two-head block passes with global rel-inf
  `1.49e-3`, zero intermediate decrypt attempts, and one final decrypt; see
  `results/runs/fhe_toy_block.json`.

OpenFHE Python is pinned in `docker/Dockerfile.openfhe`. On the Mac, the Linux x86-64
image runs under Apple Silicon emulation, so those latencies are marked `[emu]`. The
complete block ran native x86-64 CPU on Brev and is marked `[native-cpu]`.

## GPU backend

[FIDESlib 2.1.3](https://github.com/CAPS-UMU/FIDESlib) is the current performance target.
Its official feature list includes full CKKS server operations, `RotateHoisted`,
`Bootstrap`, CUDA acceleration, OpenFHE interoperability, and NCCL multi-GPU support.
Its [published evaluation](https://arxiv.org/abs/2507.04775) reports RTX 4090 primitive
times and 73.5–146 ms bootstraps for its tested slot/parameter sets. `[V, external]`

`[V, 2026-07-24]` A read-only upstream check reports
`786c7600fb2f16b724e0acf73df367b27b8afed6` as FIDESlib `HEAD`, exactly the
commit pinned by the CUDA image. There is therefore no newer upstream backend commit
to substitute before measuring packing and scheduling work:

```bash
git ls-remote https://github.com/CAPS-UMU/FIDESlib.git HEAD 'refs/tags/*'
```

That evidence selects the implementation to test; it is not a DNAGPT latency result.
The plan is C++/CUDA with a GPU-resident pass. Python remains the plaintext oracle and
evidence validator. A CPU/GPU hybrid is only a fallback if the current GPU implementation
fails this project's accuracy gate.
