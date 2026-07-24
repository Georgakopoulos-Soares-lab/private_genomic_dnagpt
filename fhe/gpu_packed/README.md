# Packed D=768/T=8 FIDESlib gate

This isolated gate is the first real-width sequence-scaling step beyond T=2.
It encrypts all eight embedded tokens in one CKKS ciphertext, evaluates block
0 LayerNorm → QKV → causal softmax attention → output projection without an
intermediate decrypt, and decrypts once for the oracle gate.

The slot contract is `copy*(1024*P)+channel*P+token`, with `P=8` and four
identical copies. Channel rotations in the 32×32 BSGS matmuls are multiplied
by `P`; token rotations are `-1..-7`. Attention is vectorized into eight causal
token offsets, so one reciprocal serves every packed query/head denominator.

`public_t8_contract.json` is `[A]` public preprocessing from the immutable T=8
fixture. Its shifts and approximation domains are fixed before encryption.
The evaluator cannot inspect a private query, adapt a bound, access a private
key, or decrypt.

Local preflight:

```bash
source .venv/bin/activate
python -m fhe.gpu_packed.packed_contract --check
python -m unittest fhe.gpu_packed.test_static_contract
```

Pinned-container compile and run:

```bash
./fhe/gpu_packed/build_in_fideslib.sh
./fhe/gpu_packed/run_packed_t8.sh \
  0 /fixture /work/evidence/fhe_fides_packed_t8_attention_TAG.json
```

The Brev launcher fails closed unless the selected physical GPU has under
100 MiB allocated memory, zero utilization, and zero compute processes. The
scheduler requires two consecutive free polls before delegating to it. Neither
script overwrites a log, completion marker, or JSON result.
