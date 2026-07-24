# Real-weight fixture and public calibration

This directory bridges the verified toy CKKS block to the released DNAGPT
`dna_gpt0.1b_m` weights. It has two deliberately separate tools:

- `export_fixture.py` creates a deterministic block-0 fixture for C++/CUDA.
- `calibrate.py` measures plaintext activation ranges on the public GSR data and
  recommends fixed approximation domains.

Neither tool modifies a checkpoint. Fixture directories are under the ignored
`checkpoints/fhe_exports/` tree by default, and every writer refuses to
overwrite an existing path.

## Frozen inputs and load contract

The tools fail before producing output unless both identities match:

| Input | Required SHA-256 |
|---|---|
| `checkpoints/classification.pth` | `d63353abdc1adba18e076b8f8b1ee9cc4d55f6948361324a825ea2245caf55f5` |
| positive GSR FASTA | `52d046d1fcf0f03bbaa4b971a1baa6b3fdbe29bc576bcdea8fe28f619efd2f11` |

The classification checkpoint must contain exactly 78 bias-free tensors:
token and position embeddings, six tensors per transformer layer, final
LayerNorm, and three classification-head tensors. Loading it into the upstream
DNAGPT class must report exactly these six unused missing tensors:

```text
number_embedding.0.weight
number_embedding.2.weight
number_embedding.3.weight
num_regression.0.weight
num_regression.2.weight
num_regression.3.weight
```

Any other missing or unexpected key is an error. This turns upstream's
permissive `strict=False` call into an explicit reproducibility contract.

## Export the real-weight block-0 fixture

From the repository root:

```bash
source .venv/bin/activate
python -m fhe.realweights.export_fixture
```

The default is the first positive AATAAA record and `T=2`. Tokenization is
exactly the released classification path:

```text
606 bp -> remove sequence[300:306] == AATAAA
prompt = <R>{600 bp sequence}<=><R>
KmerTokenizer(k=6, dynamic_kmer=True, max_len=512)
```

Use a longer prefix or float32 export explicitly:

```bash
python -m fhe.realweights.export_fixture --tokens 8 --dtype float32
```

The fixture includes:

- the six bias-free block-0 arrays: LN1, QKV, attention projection, LN2,
  MLP expansion, and MLP projection;
- token embeddings, position embeddings, and their sum (the encrypted numeric
  input boundary);
- explicit LN statistics, Q/K/V, unmasked attention scores and causal mask,
  softmax probabilities, attention contexts, both residuals, GELU input/output,
  and final block output;
- the upstream torch output and the independent NumPy reference error.

The exporter asserts that the NumPy float64 implementation matches the upstream
torch float32 block with `rtol=atol=2e-5`. On the local `T=2` smoke fixture the
observed errors were max-absolute `8.67e-6` and relative-infinity `5.50e-7`.

### Binary format

`manifest.json` is the only index. Every array entry contains its filename,
role, shape, element count, byte count, dtype, byte order, storage order, and
SHA-256. Binary files have no header:

1. read exactly `elements` IEEE-754 `float64` or `float32` values;
2. values are little-endian;
3. reshape using C row-major order and the recorded `shape`.

Linear weights retain PyTorch order `[out_features, in_features]`, so a C++
reference computes `y = x * W^T`. Q, K, and V are separately exported as
`[heads, T, head_dim]`.

## Calibrate fixed public approximation domains

Calibration uses public plaintext data, before private inference. It does not
decrypt a private ciphertext or make any data-dependent decision during an FHE
pass. A new tag is mandatory and an existing JSON is never overwritten.

Tiny smoke:

```bash
python -m fhe.realweights.calibrate \
  --tag fhe_realweight_calibration_smoke_YYYYMMDD \
  --samples 1 --positive-only --max-tokens 16 \
  --output-dir /tmp/dnagpt-calibration-smoke
```

Representative public calibration:

```bash
python -m fhe.realweights.calibrate \
  --tag fhe_realweight_calibration_gsr_128_YYYYMMDD \
  --samples 128 --max-tokens 512
```

Samples are deterministic: first positive and negative records, interleaved.
The JSON records each selected record by index and SHA-256 without embedding its
sequence. For every layer it records ranges for:

- both LayerNorm variance-plus-epsilon inputs and normalized outputs;
- finite causal attention scores and probabilities;
- GELU input and output;
- attention residual and block output.

Recommended global fixed domains expand all observed extrema by the configured
margin (default `1.25`) and round outward. They are tagged `[A]`: the domains
are calibration assumptions until validated on a broader held-out public set.

## Tests

```bash
source .venv/bin/activate
python -m unittest fhe.realweights.test_realweights
```

The fast tests do not require checkpoints or datasets. They verify the exact
key contract, prompt/token semantics, domain expansion, and an independent
NumPy-vs-upstream transformer block.
