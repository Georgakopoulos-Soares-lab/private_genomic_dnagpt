# Twelve-block T=2 plaintext/export bridge

This additive bridge extends `fhe/realweights/` from released block 0 to all 12
blocks and the GSR classifier head. It is the reproducible plaintext contract
for a future encrypted multi-block evaluator; it does not run FHE or a GPU.

The frozen public input is positive FASTA record 0,
`GL000191.1:32848-33654(+)`, sequence SHA-256
`c3f792aa4d1c08137c8aaa8e95e630c186a2676da8e0729b2b0190e2fb9ef137`.
After the released GSR transform, its first two tokens are exactly
`<R>, AATGGA` with IDs `21, 8250`. The full prompt has 103 tokens.

## Export

From the repository root:

```bash
source .venv/bin/activate
python -m fhe.multiblock.export_fixture
```

The command enforces the existing classification-checkpoint and positive-FASTA
SHA-256 contracts before producing anything. It writes a new immutable fixture
under the ignored path:

```text
checkpoints/fhe_exports/
  gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0/
```

The exporter writes:

- all six bias-free weights for each of 12 transformer blocks;
- final LayerNorm, classifier projection and classifier LayerNorm weights;
- exact `N`, `A`, and `N-A` readout rows;
- the encrypted numeric input boundary (token plus position embeddings);
- every independent float64 block intermediate and each upstream torch block
  output;
- final LayerNorm, SiLU, head LayerNorm, full logits, and an exact encrypted
  scalar readout recipe;
- per-array shapes, sizes, SHA-256 values, a canonical content identity, and
  `SHA256SUMS`.

The manual float64 lineage is gated against upstream torch float32 after every
block (`rtol=atol=2e-5`) and after the classifier head.

## Validate

Full binary hash validation:

```bash
python -m fhe.multiblock.validate_fixture \
  checkpoints/fhe_exports/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0
```

Metadata-only validation is useful for a quick local check:

```bash
python -m fhe.multiblock.validate_fixture \
  checkpoints/fhe_exports/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0 \
  --metadata-only
```

## Classification boundary

`T=2` is the smallest non-degenerate causal-attention graph gate, not a
task-representative GSR context. On this public prefix the manual plaintext
readout is `N-A = -4.6695341316` (`A` by the binary sign), while the global
vocabulary argmax is `AAAAAA`, not `N` or `A`. The complete 103-token prompt
produces `N-A = +12.2666015625`, global argmax `N`. The manifest records both,
preventing a T=2 encrypted graph result from being mislabeled as full task
accuracy.

## Public-domain evidence

For every block the manifest records both LayerNorm variance inputs and
normalized outputs, the exact T=2 encrypted-source-0 attention delta and
denominator, GELU input/output, and block output. It also records final
LayerNorm and classifier SiLU/LayerNorm ranges.

The compiled block-0 GPU domains do **not** compose through all 12 blocks.
Block 1 already exceeds the fixed LN1, attention-delta, and GELU domains.
Across this one public T=2 sample the widest ranges include:

- LN variance-plus-epsilon: up to `486.0591373`;
- attention delta: `[-20.1360437, 9.7345274]`;
- attention denominator: up to `16891.851283`;
- GELU input: `[-13.9221744, 38.3430314]`;
- block output: `[-229.2557277, 404.6620378]`;
- classifier SiLU input: `[-5.7972366, 21.9693427]`.

These ranges are `[V]` on one fixed public sample. The outward-padded domains
are `[A]` calibration assumptions and must be validated on broader public data
before any private inference. They are never selected from a decrypted private
query.

## Tests

```bash
source .venv/bin/activate
python -m unittest fhe.multiblock.test_multiblock
```

The fast tests do not need downloaded weights or data. They check the exact
72-weight contract, T=2 softmax identity, bias-free classifier reference, and
fail-closed public-domain coverage.
