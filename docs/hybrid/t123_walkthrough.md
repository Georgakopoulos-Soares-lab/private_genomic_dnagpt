# T123 walkthrough — one encrypted DNAGPT block, from input file to final decrypt

**Audience:** a programmer who is comfortable with C++/Python and shell, but who has never
worked with homomorphic encryption, GPU crypto libraries, or SLURM. Nothing here assumes
you understand lattice cryptography. Niche terms are explained the first time they appear
and collected in the [glossary](#3-glossary).

**What "T123" is:** one specific job in this repo. It takes the *encrypted* numeric
embeddings of a 103-token human-genome prompt, runs the **complete first transformer block
of DNAGPT** on them (LayerNorm → Q/K/V → causal attention → projection → residual →
LayerNorm → MLP with GELU → residual), and checks that the decrypted result matches the
plaintext PyTorch/NumPy answer. "T123" = the config-3 all-optimizations block **plus** the
Tier-1/2/3 plaintext-encoding optimizations. It is a *correctness + performance gate*, not
a product.

Everything measured below comes from one real passing run, **SLURM job 3341635**
(`kimon/logs/config3_all_opts/…_slurm_3341635/`), on TACC Lonestar6, one NVIDIA
A100-PCIE-40GB, 32 CPU cores. Tags follow the repo convention: `[V]` verified/measured,
`[A]` assumption/estimate, `[U]` unresolved.

---

## 1. The 60-second version

```
  DISK (Lustre /scratch)                    HOST RAM (CPU)                    GPU (A100, 40 GB)
  ─────────────────────                     ──────────────                    ─────────────────
  fixture *.bin  ──read──▶  weights + embeddings as double[]
  (55 MiB, float64)                              │
                                                 │ CKKS "encode"  (CPU, 32 threads)
                                                 ▼
                                          Plaintext objects  ──copy──▶  device plaintexts
                                                 │                            │
                                          Encrypt (CPU) ───────copy──────▶  ciphertexts
                                                                              │
                                            ┌───── server: all linear algebra (532 s) ─────┐
                                            │  multiply / add / rotate / key-switch        │
                                            └──────────────┬───────────────────────────────┘
                                                           │ at each nonlinearity:
                                          Decrypt ◀─copy───┘   (LayerNorm 1/√x, softmax, GELU)
                                                 │
                                          exact math in plaintext (CPU, 121 s)
                                                 │
                                          Re-encrypt ──────copy─────────▶  fresh ciphertext
                                                                              │
                                          final Decrypt ◀──copy───────────────┘
                                                 │
  result JSON + run.log + telemetry.csv ◀────write┘
```

`[V]` 663 s wall clock. `[V]` Accuracy: `global_rel_inf = 4.64e-9` against a tolerance of
`4e-2` — i.e. the encrypted answer is ~7 orders of magnitude better than required.
`[V]` Peak host RAM 48.6 GiB, peak GPU memory 9.6 GiB.

---

## 2. Cast of characters — which file does what

| Thing | Path | Role |
|---|---|---|
| **Launcher** | [kimon/configs/config3_all_opts/block_T123_slurm.sbatch](../../kimon/configs/config3_all_opts/block_T123_slurm.sbatch) | SLURM batch script. Verifies inputs, sets threads, starts a telemetry sampler, runs the binary, writes a summary. |
| **Environment** | [kimon/env/tacc_env.sh](../../kimon/env/tacc_env.sh) | `module load gcc/13.2 cuda/12.8 …`, sets `LD_LIBRARY_PATH` to the locally built FIDESlib/OpenFHE. |
| **The program** | [fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp](../../fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp) | 1762 lines of C++/CUDA. The whole encrypted block. |
| **Frozen layout contract** | [fhe/gpu_real_scheme_b/src/simd_t103_schedule.hpp](../../fhe/gpu_real_scheme_b/src/simd_t103_schedule.hpp) | Crypto-free header that *defines* the slot map and tile schedule. Its SHA-256 is pinned in the binary. |
| **Input data ("fixture")** | `checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0/` (gitignored) | Raw float64 arrays: the embeddings, the block's weights, and the expected answer. |
| **Integrity contract** | [kimon/env/fixtures_ls6/fixture_t103.sha256](../../kimon/env/fixtures_ls6/fixture_t103.sha256) | SHA-256 of every fixture file, checked before the run. |
| **Crypto backend** | `kimon/env/fideslib-src/` + `fideslib-install/` (gitignored) | FIDESlib = CUDA CKKS library, pinned at commit `786c7600…`; it wraps OpenFHE (CPU) for encode/encrypt/decrypt and does the heavy math on GPU. |
| **Micro-gate** | [fhe/gpu_real_scheme_b/microgate/clone_microtest.cpp](../../fhe/gpu_real_scheme_b/microgate/clone_microtest.cpp) | 4 small GPU tests that validated the T123 optimization in ~seconds before spending 11 minutes on the full block. |
| **Outputs** | `kimon/logs/config3_all_opts/<tag>/` and `…/slurm/T123_job_<id>/` | Result JSON, `run.log`, `telemetry.csv`, `SUMMARY.txt`. |

### 2.1 The two roles inside one process

This is the single most important structural idea. The program contains two classes:

- **`Client`** (line ~509) — plays the **data owner**. It is the only object that holds the
  secret key. Its only public methods are the *declared boundaries*:
  `invsqrt_boundary`, `gelu_boundary`, `reduce_score_tile`, `emit_weight_tile`.
- **`EncryptedEvaluator`** (line ~757) — plays the **untrusted compute provider**. It holds
  the crypto context, the plaintext weights, and a reference to the `Client`. It has *no
  access to the secret key at all*; the only way it can learn anything is by calling one of
  the four declared boundary methods.

`[V]` The result JSON records `"evaluator_has_private_key": false` and
`"intermediate_decrypt_attempts": 0`.

`[U]` **Both roles run in the same Linux process on the same node.** There is no socket, no
serialization, no key custody, and no network cost in the measured timings. The separation
is enforced by C++ visibility, which is a faithful model of *who may see what*, but not of a
deployed service. That is out of scope per `CLAUDE.md`.

Also note the threat model: **the weights are public.** The server reads `weights__*.bin`
in the clear. Only the genome-derived input is encrypted. That is what "compute provider
evaluating a model on an encrypted genome" means here.

---

## 3. Glossary

Read this once; the walkthrough leans on it.

**FHE / CKKS.** Fully Homomorphic Encryption lets you compute on ciphertexts without
decrypting. **CKKS** is the FHE scheme for *approximate* arithmetic on real numbers — think
"encrypted float vectors". It supports add, multiply, and cyclic rotation of the vector. It
does **not** support comparison, division, `exp`, `tanh`, or `1/√x`.

**Slot / SIMD packing.** One CKKS ciphertext does not hold one number; it holds a *vector* of
`SLOTS` numbers, and every operation is elementwise across all of them at once — like an
enormous SIMD register. Here `SLOTS = 32768`. Doing one multiply on a 32768-slot ciphertext
costs the same as doing it on one number, so the entire optimization game is "arrange the
data so the slots are busy".

**Ring dimension (`RING_DIM = 65536`).** The polynomial degree `N` underlying the scheme. For
real-valued data you get `N/2 = 32768` usable slots. Bigger `N` = more security and more
slots, but every operation costs more and every ciphertext is bigger.

**RNS limb / tower.** A ciphertext coefficient is a huge integer (hundreds of bits). Rather
than use bignums, CKKS represents it in **Residue Number System** form: a list of remainders
modulo a chain of ~50-bit primes. Each entry is a "limb" or "tower". One limb here is
`65536 × 8 bytes = 512 KiB`. Practical consequence: **ciphertext and plaintext size is
proportional to the number of limbs**, and the number of limbs shrinks as you consume depth.

**Level / multiplicative depth / rescaling.** Every multiplication inflates the encrypted
scale, so CKKS "rescales" — drops one limb — to bring it back. That limb is gone forever.
`MULT_DEPTH = 13` means the moduli chain permits 13 such drops. **Level** = how many you have
already spent. Level 13 = out of budget; the ciphertext must be refreshed (bootstrapped, or —
in our architecture — decrypted and re-encrypted by the client) before it can be multiplied
again. Think of it as a battery that only multiplications drain. `[V]` This run finishes the
block at level 6 of 13.

**Scale bits (`SCALE_BITS = 50`, `FIRST_MOD_BITS = 60`).** How many bits of fixed-point
precision each level carries. Total modulus ≈ 60 + 13×50 = 710 bits, which at `N = 65536`
still satisfies the 128-bit security level (`HEStd_128_classic`).

**Encode vs Encrypt.** Two distinct steps, and the distinction is the whole point of T123:
1. **Encode** — turn a `double[32768]` into a `Plaintext`: an inverse-DFT-like transform plus
   RNS conversion. Pure CPU, no key involved, and *expensive* (tens of ms).
2. **Encrypt** — turn a `Plaintext` into a `Ciphertext` using the public key.

You can multiply a ciphertext by a *plaintext* (`ct × pt`) or by another *ciphertext*
(`ct × ct`). The rule that decides which you get is **not** "cheap vs expensive", it is
**who the operands belong to**:

- **`ct × pt`** — one operand is a *public constant*: a model weight, a mask, a scale factor.
  Anyone may know it, so it never needs encrypting. Cheap: no key switching.
- **`ct × ct`** — *both* operands are derived from the user's genome, so both must stay
  encrypted. `[A]` Several times more expensive, because the product is encrypted under a
  "squared" key and needs an extra **relinearization** key switch to become a normal
  ciphertext again.

`[V]` This run does 177,734 `ct × pt` and only 1,506 `ct × ct`.

**Rotation / rotation key.** Cyclically shifting the 32768 slots by `k`. This is how you move
data between lanes, and it is the only "cross-slot" primitive. Each distinct shift amount
needs its own pre-generated **rotation key** (a few hundred MB total). `[V]` This run
generates 80 rotation keys and performs 8,173 rotations.

**Key switching / HYBRID / digits (`LARGE_DIGITS = 3`).** After a `ct × ct` multiply or a
rotation, the ciphertext is encrypted under the "wrong" key and must be switched back using a
key-switching key. HYBRID key switching splits the operand into `dnum = 3` digits — a
three-way tradeoff between speed, memory for the keys, and noise growth. Practical
consequence: this is where most GPU time goes.

**BSGS (baby-step / giant-step).** The trick for multiplying an encrypted vector by a public
`768×768` matrix. Naively you would need 768 rotations. Instead you decompose the matrix into
its 1024 *diagonals* and factor the rotation set as `32 × 32`: rotate the input 31 times
("baby steps", reused for all diagonals), and rotate partial sums 31 times ("giant steps").
This is exactly loop tiling/blocking, applied to rotations. `[V]` Cost per matrix: 1024
`ct × pt` multiplies + ~62 rotations instead of 768 rotations.

**Client boundary (this project's architecture).** CKKS cannot evaluate `1/√x`, `softmax`, or
`GELU`. Rather than approximate them with polynomials (lossy, depth-hungry), we send that one
ciphertext to the client, who decrypts it, evaluates the function *exactly* in float64,
re-encrypts, and sends it back. The server never sees plaintext and never holds the key; the
client only ever sees values derived from its own query. This is the "client-assisted CKKS"
(legacy tag *Scheme B*) architecture. Side benefit: re-encryption resets the level to 0, so
each boundary is also a free depth refresh.

**Fixture / oracle / gate.** *Fixture* = the frozen input data directory. *Oracle* = the
plaintext answer the encrypted run must reproduce (`oracle__block_output.bin`, produced by an
independent NumPy float64 reimplementation that was itself checked against upstream PyTorch
DNAGPT). *Gate* = a pass/fail test; this job passes iff the decrypted output matches the
oracle within tolerance **and** every operation count equals its hard-coded expected value.

### 3.1 Aside: if the weights are public, why is there any `ct × ct` at all?

Because a transformer does not only multiply **data × weights**. It also multiplies
**data × data**, and *that* is what forces `ct × ct`. Every model weight in this run really is
a plaintext — no exceptions — so all 156 matrix products are `ct × pt`. The 1,506 `ct × ct`
multiplies come from exactly four places in the source, and none of them involves a weight:

| Source | Expression | Why both sides are encrypted | Count |
|---|---|---:|---:|
| `layernorm()`, line 1172 | `centered × centered` | Squaring the centered input to get the variance. Both operands *are* the genome-derived activation. | 26 |
| `layernorm()`, line 1176 | `centered × inverse` | `inverse` = `1/√variance`, which the **client** computed from its own data and **re-encrypted**. Handing it back in the clear would leak the query. | 26 |
| score stage, line 808 | `query × shifted_key` | The `QKᵗ` product. Q and K are both encrypted projections of the input. This is the irreducible data×data step of attention. | 727 |
| context stage, line 851 | `attention_weight × shifted_value` | The `softmax(·) × V` product. The attention weight came back from the client re-encrypted; V is encrypted. | 727 |
| | | **total** | **1,506** |

`[V]` 26 + 26 + 727 + 727 = 1,506 = the pinned `EXPECTED_CT_CT` invariant, so the accounting
is exact — there is no fifth hidden source.

Three things worth taking from that table:

1. **Attention is inherently data×data.** A plain feed-forward network on public weights would
   need *zero* `ct × ct`: every step is activation × constant. Attention compares one part of
   the input against another part of the same input (`QKᵗ`, then `softmax × V`), and no amount
   of clever packing makes those operands public. `[V]` This is why 1,454 of the 1,506 `ct × ct`
   (97 %) live in the attention stages.
2. **LayerNorm needs a square and a rescale.** `variance = mean((x − μ)²)` is a squaring, and
   the final `centered × inverse` divides by a *data-dependent* quantity. Division by a public
   constant would be `ct × pt`; division by your own standard deviation cannot be.
3. **Anything the client returns comes back as a ciphertext, by construction.** The client
   evaluates `1/√x` and softmax in the clear *on its own side*, but it must re-encrypt before
   returning, or the untrusted server would receive plaintext derived from the genome — which
   is the one thing this architecture exists to prevent. So a client boundary converts a
   would-be `ct × pt` into a `ct × ct`. `[V]` That is the source of `26 + 727 = 753` of them,
   i.e. **half the `ct × ct` count is the price of the client-assisted design itself**, not of
   the model.

Terminology trap: the variable at line 851 is named `weight` in the source and the word
"weight" appears in `emit_weight_tile` / `weight_tile_encryptions`. Throughout the attention
stages that means **attention weight** — a softmax probability — never a model parameter. The
snippets in §5c below spell it `attn_weight` to keep the two apart.

---

## 4. The data layout — how 103 tokens fit in 32768 slots

DNAGPT-0.1b block dimensions: `D = 768` features, `HEADS = 12`, `HEAD_DIM = 64`,
`MLP_DIM = 3072 = 4 × D`. The prompt is `T = 103` tokens.

Three nested ideas:

1. **`PACK_WIDTH = 1024`** — a 768-feature vector is padded to a 1024-wide power-of-two
   region so BSGS rotations wrap cleanly (`32 × 32 = 1024`).
2. **`COPIES = 4`** — four copies of that region sit side by side, because the MLP hidden
   layer is `4 × 768`. Copy `c` computes the `c`-th quarter of the MLP.
3. **`TOKEN_BATCH = 8`** — eight tokens are interleaved in the *innermost* dimension, so 8
   tokens are processed per ciphertext.

`4 × 1024 × 8 = 32768 = SLOTS`. The address function (source line 419):

```cpp
physical_slot(copy, feature, token_lane) = (copy * PACK_WIDTH + feature) * TOKEN_BATCH + token_lane
```

```
slot index:   0    1    2  ...  7  |  8    9   ... 15  | ...
             ─────────────────────────────────────────────────
copy 0, feature 0: tokens 0..7     |
copy 0, feature 1:                   tokens 0..7      |
...
copy 0, feature 767 (then 768..1023 = padding)
copy 1, feature 0: tokens 0..7 (same 8 tokens, second MLP quarter)
...
```

Because tokens are *innermost*, every "logical" rotation by `k` features becomes a physical
rotation by `8 × k`, which structurally **cannot** leak data across token lanes. And every
public coefficient must be repeated across the 8 innermost lanes — that is what all those
`for (token = 0; token < TOKEN_BATCH; ++token)` loops in the plaintext builders are doing.

`T = 103` tokens ÷ 8 = **13 token groups** (12 full groups + a last group with 7 active lanes;
`static_assert(T % TOKEN_BATCH == 7)`). Almost the whole program is a loop over those 13
groups.

Payoff `[V]`: the frozen one-token-at-a-time schedule needs `12 × 103 = 1236` matrix
products; this run needs **156** (= 13 groups × 12 distinct weight matrices). The JSON calls
it `"dense_call_reduction": 7.92`.

---

## 5. Walkthrough

### Phase 0 — Submission (login node, seconds)

```bash
sbatch kimon/configs/config3_all_opts/block_T123_slurm.sbatch
```

SLURM queues the job (`-p gpu-a100-small`, 1 node, 32 cores, 3 h limit) and later runs the
script *as an ordinary bash script* on a compute node. The `#SBATCH` lines are just comments
to bash — the same file runs on any GPU box with `bash`.

### Phase 1 — Pre-flight on the compute node (~1 s, CPU only)

The script, in order:

1. `mkdir` the evidence directory; `source kimon/env/tacc_env.sh` → `module load gcc/13.2.0
   cuda/12.8 …`, `LD_LIBRARY_PATH` → the locally built FIDESlib/OpenFHE shared objects.
2. **Force `OMP_NUM_THREADS`.** TACC pre-sets it to 1; the script overrides it to the node's
   core count (`[V]` 32) because the T123 encode loop is OpenMP-parallel. It also pins threads
   (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`) — speed only, numerics unchanged.
3. Assert the binary exists and is executable; assert the fixture manifest exists.
4. `nvidia-smi` → record which GPU we got.
5. **`sha256sum --check` the whole fixture** against `fixture_t103.sha256`. If a single input
   byte differs, the job aborts here. This is why a result can be trusted later.
6. Compute SHA-256 of the source, its parent source, the frozen semantic anchor, the schedule
   header, the fixture manifest, and the contract — and pass them all as CLI flags. **The
   binary re-checks them against values compiled into it** and refuses to run on a mismatch
   (source lines 317–335). Provenance is enforced by the program, not by trust.
7. Start a **background telemetry sampler** — a subshell that every 5 s appends GPU memory,
   GPU utilization, the target process's `VmRSS`/`VmHWM` from `/proc/<pid>/status`, host load
   average, and (newer version) CPU-cores-busy from `/proc/<pid>/stat`, to `telemetry.csv`.
   This is an *independent* measurement of the program's own timers.

### Phase 2 — Load the fixture (`[V]` 0.045 s, CPU + host RAM, reads disk)

`load_fixture()` reads headerless little-endian float64 arrays with `std::ifstream`. Each
read validates size and finiteness and throws otherwise.

| File | Shape | Bytes | What it is |
|---|---|---|---|
| `input__embeddings.bin` | 103 × 768 | 632,832 | **The encrypted-side input**: token + position embeddings from the tokenized GSR prompt. |
| `weights__ln1.bin`, `weights__ln2.bin` | 768 | 6,144 each | LayerNorm gains (DNAGPT-0.1b has no LN bias). |
| `weights__attn_qkv.bin` | 3 × 768 × 768 | 14,155,776 | Fused Q, K, V projections; split into 3 matrices in RAM. |
| `weights__attn_proj.bin` | 768 × 768 | 4,718,592 | Attention output projection. |
| `weights__mlp_fc.bin` | 3072 × 768 | 18,874,368 | MLP up-projection; split into 4 × (768×768). |
| `weights__mlp_proj.bin` | 768 × 3072 | 18,874,368 | MLP down-projection; **de-interleaved** into 4 × (768×768) to match the COPIES layout. |
| `oracle__block_output.bin` | 103 × 768 | 632,832 | The answer. Never used by the evaluator — only by the final metric. |

`[V]` ≈ 55 MiB read from Lustre, held as `std::vector<double>` in host RAM for the whole run.

Worth internalizing: **the 12 `std::vector<double>` weight matrices keep one fixed address for
the entire run**, and the T123 caches are keyed by that address. That is a deliberate design
choice (`&weight` as a stable identity), documented at source line 1205.

### Phase 3 — Crypto context, key generation, GPU load (`[V]` 4.65 s)

```cpp
parameters.SetSecurityLevel(HEStd_128_classic);   // 128-bit classical security
parameters.SetMultiplicativeDepth(13);            // 13 levels of budget
parameters.SetScalingModSize(50);                 // 50-bit scale per level
parameters.SetRingDim(65536);                     // N; 32768 real slots
parameters.SetScalingTechnique(FLEXIBLEAUTO);     // library manages rescaling
parameters.SetKeySwitchTechnique(HYBRID);
parameters.SetNumLargeDigits(3);
parameters.SetDevices({options.gpu});             // which physical GPU
parameters.SetPlaintextAutoload(false);           // <-- important, see §7
parameters.SetCiphertextAutoload(true);
```

Then `KeyGen()` (secret + public key), `EvalMultKeyGen()` (relinearization key for `ct × ct`),
`EvalRotateKeyGen(rotation_keys)` — `[V]` 80 keys, computed by `required_rotation_keys()` as
the union of: 31 BSGS baby steps, 31 giant steps, the log-style accumulate-sum ladder, the 14
token-lane shifts, and 3 cross-copy shifts.

**`cc->LoadContext(publicKey)` is where the GPU wakes up.** FIDESlib builds its CUDA context
(NTT twiddle tables, prime tables) and uploads the evaluation and rotation keys to device
memory. `cc->Synchronize()` blocks until the CUDA stream drains — necessary because otherwise
the timing numbers would be meaningless.

`[V]` Telemetry: GPU memory jumps 1 MiB → 5,731 MiB → 7,791 MiB in the first ~10 s. That is
almost entirely **key material resident on the GPU for the whole run**.

`FLEXIBLEAUTO` note: the library decides *when* to rescale (it defers), so `GetLevel()`
reflects rescales actually performed, not a naive multiply count. Don't hand-derive levels
from the source; read them off the log (`[stage] … level=`).

### Phase 4 — Encrypt the input (`[V]` 1.96 s)

For each of the 13 token groups (source line 1642):

1. **CPU:** build `double packed[32768]`, writing `fixture.input[token*768 + dim]` into
   `physical_slot(copy, dim, lane)` for all 4 copies. (Replicated 4× up front so the MLP
   stage doesn't have to broadcast later.)
2. **CPU:** `MakeCKKSPackedPlaintext(...)` — encode.
3. **CPU→GPU:** `Encrypt(publicKey, plaintext)` — and because `CiphertextAutoload = true`,
   the resulting ciphertext is uploaded to the device.

`[V]` 13 ciphertexts, each ~14 limbs × 2 polynomials × 512 KiB ≈ 14 MiB on the GPU `[A]`.
This is the **only** input crossing the trust boundary. Encrypted token-*index* lookup (the
tokenizer itself) is explicitly `[U]` out of scope.

### Phase 5 — The encrypted block (`[V]` 652.5 s = 98 % of wall clock)

`EncryptedEvaluator::evaluate()`. Five stages; per-stage behaviour was reconstructed from the
`run.log` stage prints plus `telemetry.csv`.

#### 5a. LayerNorm 1 + Q/K/V — 13 groups (`[V]` ~0–115 s, GPU util ~38 %, RSS → 25.5 GiB)

`layernorm()` (line 1168) is a nice illustration of the whole architecture:

```
mean      = AccumulateSum(x) × (1/D)          GPU: rotate+add ladder, then ct×pt (1/D is public)
centered  = x − mean                          GPU: free (addition costs no level)
variance  = AccumulateSum(centered²) × (1/D)  GPU: ct×ct — squaring the data (§3.1)
variance += 1e-5                              GPU: free
inverse   = CLIENT( 1/√variance )             ◀── boundary: decrypt, exact float64, re-encrypt
normalized = centered × inverse               GPU: ct×ct — `inverse` came back encrypted
output     = normalized × ln_weight           GPU: ct×pt — ln_weight is a public model weight
```

Note the last two lines side by side: they are the clearest small example of the rule in §3.1.
Multiplying by the LayerNorm *gain* is `ct × pt` because that gain is a public model parameter;
multiplying by `1/√variance` is `ct × ct` because that number is a property of the user's own
genome.

`AccumulateSum(input, 1024, 8)` is the sum-and-broadcast over the 1024-wide feature region,
implemented as a base-4 rotate-and-add ladder — that is what `add_accumulate_keys()`
pre-generates keys for. `[V]` 8,776 calls total.

The `invsqrt_boundary` (line 514) is worth reading closely, because all four boundaries have
the same five-step shape:

```cpp
cc_->Decrypt(keys_.secretKey, local, &plaintext);            // GPU→CPU copy, then CPU decrypt
plaintext->SetLength(SLOTS);
std::vector<double> values = plaintext->GetRealPackedValue(); // 32768 doubles on CPU
for (double& v : values) v = exact_invsqrt(v);               // exact 1/sqrt in float64
Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);
Ct output = cc_->Encrypt(keys_.publicKey, refreshed);        // fresh level-0 ciphertext
```

Note `1/√x` is applied to **all** 32768 slots including padding — cheap and harmless, because
only the meaningful slots are ever read downstream.

Then `baby_rotations()` produces the 31 BSGS baby steps in one batched
`EvalFastRotation` call (GPU), and `matmul()` runs 1024 `ct × pt` multiplies + 31 giant-step
rotations, three times (Q, K, V).

`[V]` Log: `[stage] qkv group=0 active=8 level=3` — after LN1 + one matrix, we have spent 3 of
13 levels.

Then — and this is a T123-specific line (source 793):

```cpp
flush_diagonal_plains();   // QKV weights are done; free their encoded templates
```

`[V]` Telemetry confirms it: RSS drops 25,530 → 12,434 MiB at t≈125 s.

#### 5b. Attention scores — 91 causal tiles (`[V]` ~125–330 s, GPU util 15–23 %, host load → 30)

13 query groups × 13 key groups, but only `query_group ≥ key_group` is needed (causal
masking): `13 × 14 / 2 = 91` tiles. `[V]` JSON: `score_tile_decryptions: 91`.

Within a tile, the 8 query lanes must be dotted against the 8 key lanes. Since both live in
the *same* innermost lane dimension, the code rotates the key ciphertext through the 8 lane
offsets (`lane_shift`, `build_group_shifts`) and, for each needed offset `delta`:

```
products = query × shifted_key                     GPU: ct×ct
for head in 0..11:
    masked   = products × head_mask(head)          GPU: ct×pt  (mask encoded ON THE SPOT)
    score    = AccumulateSum(masked)               GPU: rotate+add ladder → the dot product
    isolated = score × score_isolation(...)        GPU: ct×pt, folds in 1/√64 and the causal mask
    packed_score += isolated                       GPU: free
CLIENT.reduce_score_tile(packed_score, q, k)       ◀── boundary: decrypt, harvest the scores
```

`reduce_score_tile` decrypts and copies the raw scores into a host-side
`scores_[token][head][token]` array. **Nothing is computed there yet** — softmax needs a
whole row, and a row spans several tiles.

`[V]` This is the stretch where GPU utilization drops to 15–23 % while host load average
climbs to ~30. Diagnosis: `head_mask_plain` and `score_isolation_plain` are still
**encoded fresh on every use** — 727 (tile,delta) combinations × 12 heads × 2 masks ≈ 17.4 k
CPU encodes that T123 does *not* optimize. `[U]` That is the next obvious Tier-1 target.

After all 91 tiles, `finalize_attention()` runs the **numerically stable softmax** entirely on
the CPU in float64: per (row, head), subtract the row max, `exp`, normalize by the sum over
`col ≤ row`. It throws if any score is non-finite or missing — a real guard against a silently
skipped tile.

#### 5c. Attention context — 727 weight tiles (`[V]` ~330–460 s, GPU util 1–2 %)

Now the reverse direction: the client *encrypts* the softmax weights and the server multiplies
them into the values.

```
for key_group, for query_group ≥ key_group, for each needed delta:
    attn_weight  = CLIENT.emit_weight_tile(q, k, delta)   ◀── boundary: pack + encrypt
    contribution = attn_weight × shifted_value            GPU: ct×ct  (softmax prob × V)
    context[q]  += contribution                           GPU: free
```

`attn_weight` is a **softmax probability**, not a model parameter (§3.1) — and it arrives as a
*ciphertext* because the client must re-encrypt before returning it, which is why this line is
`ct × ct` rather than `ct × pt`.

`emit_weight_tile` broadcasts each scalar attention weight across that head's 64 feature slots
and all 4 copies, then encodes + encrypts. `[V]` 727 tiles → 727 encode+encrypt pairs on the
CPU. This is the most **client-heavy, GPU-idle** part of the run: telemetry shows
`gpu_util_pct` at 1–2 % for roughly 90 s.

`[V]` `client_boundary_seconds_total = 120.6 s` — 18.5 % of the encrypted evaluation. That is
the measured price of the client-assisted architecture, and it is dominated by these 727
encrypt-and-return round trips plus 91 decrypts.

`[V]` `round_trips = 857 = 26 (LN) + 13 (GELU) + 91 (scores) + 727 (weights)`, and
`logical_boundary_instances = 129,162`. The latter is an implementation schedule counter with
mixed units: LayerNorm counts active tokens, GELU counts token-copies, and attention counts
head/query/key scalars. It is auditable but is not a homogeneous scalar-work or traffic count.

#### 5d. Attention projection + residual (fast)

13 × `matmul(baby_rotations(context[g]), attention_projection)`, then
`residual1[g] = encrypted_inputs[g] + attention_projection[g]`. The residual reaches back to
the **original level-0 input ciphertexts**, which are still resident — a genuine ~14 MiB × 13
of GPU memory held for ~10 minutes just to add them at the end. Then another
`flush_diagonal_plains()`.

#### 5e. LayerNorm 2 + MLP with GELU (`[V]` ~460–655 s, GPU util ~37 %, RSS → 48.6 GiB peak)

Per token group:

1. `layernorm(residual1[g], …, "ln2")` — second client boundary of this group.
2. `require_remaining_depth(level, 4, …)` — an explicit assertion that ≥ 4 levels remain.
   Better to abort with a clear message than produce noise.
3. Four `matmul` calls against `mlp_fc[0..3]` → the 3072-wide hidden layer, one quarter per
   COPY. This is where the `COPIES = 4` layout pays off.
4. Mask each quarter with `copy_active_plain(chunk)` and add → one ciphertext holding all 3072
   hidden values.
5. **`gelu_boundary`** — decrypt, apply the exact tanh-form GELU to the `4 × 768 × active`
   meaningful slots in float64, re-encrypt. `[V]` 13 calls.
6. Re-split by copy, `replicate_copies()` (3 cross-copy rotations + adds) so each quarter is
   broadcast where the down-projection needs it.
7. Four `matmul` calls against `mlp_projection[0..3]`, summed → the MLP output.
8. `block_output[g] = residual1[g] + mlp`.

`[V]` Log: `[stage] block output group=12 level=6` — the block finishes at level 6 of 13, with
7 levels of headroom. That headroom is what makes multi-block composition plausible.

`[V]` Peak host RSS 48.6 GiB happens here: the MLP stage keeps **8** encoded weight-template
sets resident simultaneously (§7 explains why, and why it is not 12).

### Phase 6 — Final decrypt (`[V]` 0.64 s)

13 `Decrypt` calls (GPU→CPU copy + CPU decrypt), then read `physical_slot(0, dim, lane)` for
each active lane and scatter into `output[103][768]`. Only **copy 0** is read — the other
three copies hold the same data and exist purely for the MLP.

### Phase 7 — Verification, and only then evidence

Two independent checks, in this order:

1. **Schedule invariants** (source line 1698). Every operation count must equal a hard-coded
   constant:

   | Counter | Expected | Meaning |
   |---|---|---|
   | `matrix_products` | 156 | 13 groups × 12 weight matrices |
   | `ct_ct_multiplications` | 1,506 | data×data only: `26 + 26 + 727 + 727` (§3.1) |
   | `ct_plain_multiplications` | 177,734 | of which `156 × 1024 = 159,744` are BSGS diagonals |
   | `explicit_rotations` | 8,173 | |
   | `accumulate_sum_calls` | 8,776 | |
   | `score_tile_decryptions` | 91 | causal tiles |
   | `weight_tile_encryptions` | 727 | |
   | `cached_key/value_lane_shifts` | 90 each | |
   | `round_trips` | 857 | client boundary crossings |
   | `logical_instances` | 129,162 | heterogeneous schedule counter; not a traffic total |
   | `diagonal_cache_misses` / `hits` | 12 / 144 | 12 distinct weights; 156 − 12 reuses |

   A mismatch throws — the run fails even if the numbers are accurate. This is what caught the
   bug in job 3341462 (§9).

2. **Accuracy** vs the oracle: max relative ∞-norm error globally and per token.
   `[V]` `global_rel_inf = 4.64e-9`, `worst_token_rel_inf = 4.70e-9`,
   `max_abs_error = 7.44e-8`, all finite, `tol = 4e-2` → **PASS**.

   That 9-digit agreement is the point of the architecture: because every nonlinearity is
   evaluated exactly rather than by polynomial approximation, the only error source is CKKS's
   own approximate arithmetic.

### Phase 8 — Write evidence to disk

`write_exclusive()` uses `open(..., O_WRONLY | O_CREAT | O_EXCL)`. `O_EXCL` fails if the file
exists — a syscall-level guarantee that an immutable result is never silently overwritten (the
option parser also refuses up front). Files produced:

| Path | Written by | Content |
|---|---|---|
| `<tag>/<tag>.json` | the C++ binary | The result. Config, all op counts, all timings, accuracy, `passed`. |
| `<tag>/run.log` | shell redirect | Full stdout/stderr, ending in `…_T123_GATE_PASS`. |
| `slurm/T123_job_<id>/telemetry.csv` | the sampler subshell | 5 s samples of GPU/RSS/load. |
| `slurm/T123_job_<id>/SUMMARY.txt` | the sbatch script | Grepped highlights + independently measured wall clock, GPU-memory peak, RSS peak. |
| `slurm/T123-<id>.out/.err` | SLURM | The script's own output. |

The script then exits 0 only if the binary exited 0 **and** the JSON contains
`"passed": true`.

---

## 6. Where does each thing actually run?

| Work | Component | Notes |
|---|---|---|
| Reading fixture `.bin` | **Disk → host RAM**, CPU | 55 MiB, once, 0.045 s |
| Plaintext weights (`vector<double>`) | **host RAM**, CPU | ~55 MiB, resident whole run |
| CKKS **encode** (`MakeCKKSPackedPlaintext`) | **CPU**, OpenFHE, OpenMP-parallel in T123 | The optimization target. ~tens of ms each. |
| **Encrypt** / **Decrypt** | **CPU** (OpenFHE) + a PCIe copy | Only the `Client` does the key-bearing ones |
| Encoded plaintext templates | **host RAM** | `[V]` up to 48.6 GiB; §7 |
| Public/rotation/relin keys | **GPU VRAM**, whole run | `[V]` ~7.8 GiB after `LoadContext` |
| Ciphertexts | **GPU VRAM** (`CiphertextAutoload = true`) | `[V]` peak total GPU 9.6 GiB |
| `EvalMult`, `EvalAdd`, `EvalRotate`, key switching, NTTs | **GPU SMs** | `[V]` 532 s of 653 s |
| `1/√x`, softmax, GELU | **CPU**, float64, exact | `[V]` 121 s of 653 s |
| Softmax score/weight tables | **host RAM** | `103 × 12 × 103` doubles ≈ 1 MiB × 2 |
| Telemetry sampling | **CPU**, separate subshell | `nvidia-smi` + `/proc` every 5 s |
| Result JSON / logs | **Disk** | `[V]` ~10 KiB total |

`[V]` GPU utilization never exceeds ~40 % and drops to 1–2 % during the client-heavy
attention-context stage. `[U]` The A100's compute is not the bottleneck in this
configuration; host-side encode work and the serialized client boundaries are.

---

## 7. What T123 actually changed, and the memory tradeoff

The parent implementation rebuilt **and re-encoded** the packed diagonal plaintext on every
single BSGS multiply: `156 matmuls × 1024 diagonals = 159,744` encodes, of which ~90 % were
byte-for-byte identical (each of the 12 weight matrices is reused across all 13 token groups).
Two clean-host repeats of the parent measured the process pegged at 3900–4700 % CPU under
enormous host load — **the CPU encode path, not the GPU, was the wall.**

T123 has three tiers:

- **Tier 1 — encode once, clone per use.** Each distinct diagonal is encoded once into a
  pristine, *unloaded* `Plaintext` template. Each multiply gets `clone_encoded(tmpl)`: a new
  `PlaintextImpl` that **shares the encoded CPU data** (a `std::any`/`shared_ptr` copy — no
  re-encode) but has fresh device state (`loaded = false`, `gpu = 0`).
- **Tier 2/3 — parallel batched encode.** The 1024 templates for a weight are encoded in one
  `#pragma omp parallel for schedule(dynamic)` across `[V]` 32 cores.

**Why the clone, and not just reuse the `Plaintext`?** This is the best cautionary tale in the
codebase. An earlier attempt (2026-07-27/28) cached the `Plaintext` *object* and reused it
across multiplies. It crashed 6/6 real-GPU attempts, root-caused with `addr2line` to
`Ciphertext::multPt → adjustPlaintextToCiphertext → RNSPoly::grow → GPUmalloc`. Mechanism:
FIDESlib's `LoadPlaintext()` returns early when `pt->loaded` is set, so a template that was
already loaded at one level was silently reused at a different level with stale GPU limbs. The
clone sidesteps this structurally: `loaded = false` forces a fresh device load at the correct
level, and `~PlaintextImpl` only evicts when `(loaded && gpu != 0)`, so the lifecycle is
per-op identical to the parent's. `[V]` Validated in isolation by
[microgate/clone_microtest.cpp](../../fhe/gpu_real_scheme_b/microgate/clone_microtest.cpp)
(TEST_A correctness, TEST_B cross-level reuse, TEST_C thousands of clones with stable GPU
memory, TEST_D parallel-vs-serial encode equality) **before** touching the 11-minute driver.

**The cost is host RAM.** 1024 templates × ~6–8 MiB each ≈ 6–8 GiB **per weight matrix**
`[A]` (a plaintext at level ℓ is `(14 − ℓ)` RNS limbs × 512 KiB). Two mitigations, both
visible in the code and in the telemetry:

1. **Encode at the use-level, not level 0.** `cached_diagonal_plains(weight, level)` is keyed
   by `(weight address, level)` and encodes with the target level, so templates carry fewer
   limbs and are roughly half the size. This is exact, not an approximation — OpenFHE would
   drop a level-0 plaintext to that level anyway.
2. **Flush between stages.** `flush_diagonal_plains()` after QKV and after the attention
   projection. Within a stage all its weights must stay resident (the loop is group-major, so
   reuse happens *across* the 13 groups), but no weight recurs across stages. So the peak is
   bounded by the largest stage — the MLP's 8 matrices — instead of all 12 at once.

`[V]` Measured: 25.5 GiB (QKV, 3 weights) → 12.4 GiB (attention, 0 weights) → 48.6 GiB (MLP,
8 weights). The shape of that curve is the design, confirmed.

---

## 8. The measured trace (job 3341635)

`[V]` Timings (from the result JSON — all measured with `std::chrono::steady_clock`):

| Phase | Seconds | Share |
|---|---:|---:|
| Load verified fixture | 0.045 | 0.0 % |
| Context + keygen + GPU load | 4.653 | 0.7 % |
| Encrypt 13 input ciphertexts | 1.956 | 0.3 % |
| **Encrypted evaluation** | **652.545** | **98.4 %** |
|  ├ provider-side encrypted evaluation (GPU backend + host support) | 531.933 | 80.2 % |
|  └ client boundaries (CPU, exact) | 120.612 | 18.2 % |
| Final decrypt (13 ciphertexts) | 0.644 | 0.1 % |
| **Wall clock (measured by the shell)** | **663** | |

`[V]` Telemetry, thinned to every 40 s:

| t (s) | GPU MiB | GPU % | RSS MiB | load1 | stage |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 0 | 11 | 0.8 | process start |
| 42 | 7,791 | 39 | 25,529 | 5.8 | LN1 + QKV, templates building |
| 83 | 8,815 | 38 | 25,530 | 3.3 | QKV |
| 125 | 8,815 | 23 | 12,434 | 11.2 | **flush**, score tiles begin |
| 168–296 | 8,815 | 15–23 | 12,434 | 22–30 | score tiles (mask encodes on CPU) |
| 339–422 | 8,815 | 1–2 | 11,557 | 22→10 | **client weight tiles — GPU idle** |
| 464 | 8,815 | 40 | 18,141 | 5.4 | attention projection + residual |
| 506–630 | 9,839 | 34–40 | 48,610 | 3.3→1.3 | LN2 + MLP + GELU (**RAM peak**) |
| 662 | 9,839 | 0 | 20,285 | 1.2 | final decrypt, teardown |

Two conclusions a beginner should take from this table: (1) the crypto is *not* GPU-bound
here; (2) the memory curve is a direct readout of the caching strategy.

---

## 9. The two failures before the pass — and what they teach

Both are real jobs whose logs are in the repo.

**Job 3341433 — killed by host memory exhaustion.** `[V]` RSS peaked at 61,863 MiB and
`T123-3341433.err` reads `fork: Cannot allocate memory`. `run.log` stops right after
`[boundary] ln2 logical=8`, i.e. the instant the MLP stage started building its templates on
top of everything still cached from earlier stages. This is what motivated the stage flushes
and the encode-at-use-level change. Lesson: on a shared HPC node, *host* RAM is as much a hard
wall as GPU VRAM, and a cache that is a pure win for time can be fatal for space.

**Job 3341462 — `error: Token-SIMD complete-block schedule count mismatch`.** `[V]` Exit code
2 after producing every `[stage] block output` line. The math ran to completion; the
*bookkeeping* didn't match. When the cache moved from once-per-diagonal to once-per-`matmul`
lookups, the expected hit/miss counts changed (they are now `12` misses / `144` hits). The
invariant refused to let a run be recorded whose op accounting hadn't been re-derived. Lesson:
this is the guardrail working. A run that "produced a number" is not a result.

**Job 3341635 — PASS.** `[V]` `gate = PASS`, `passed_json=yes`, exit 0.

---

## 10. Running it yourself

```bash
# from the repo root, on a Lonestar6 login node
sbatch kimon/configs/config3_all_opts/block_T123_slurm.sbatch
squeue -u $USER

# after it finishes
cat kimon/logs/config3_all_opts/slurm/T123_job_<jobid>/SUMMARY.txt
```

Off SLURM (any A100-class box with the native FIDESlib build and the fixture):

```bash
bash kimon/configs/config3_all_opts/block_T123_slurm.sbatch      # #SBATCH lines are just comments
KIMON_OMP=16 bash kimon/configs/config3_all_opts/block_T123_slurm.sbatch   # cap encode threads
```

Reading a result: `passed` is the verdict; `global_rel_inf` is the accuracy;
`timings_seconds.server_linear_algebra_seconds` vs `client_boundary_seconds_total` is the
architecture split; `diagonal_cache_hits/misses` proves the optimization engaged;
`SUMMARY.txt`'s `target_rss_peak_mib` is your host-memory headroom check.

If it fails: `[FATAL] fixture arrays do not match…` = wrong/corrupt fixture;
`refusing unpinned…` = a source or fixture SHA changed, so re-derive the pins deliberately;
`refusing to overwrite immutable evidence` = that tag already exists, use a new job;
`schedule count mismatch` = an op count changed, so re-derive the invariant before believing
anything; `fork: Cannot allocate memory` = host RAM, lower `KIMON_OMP` or check the flush
points.

---

## 11. Honest limits of this job

- `[V]` This is **1 of the 12 DNAGPT blocks**, plus no task head. The 12-block + GSR-head
  driver exists but has not been run at T=103 (`CLAUDE.md`, `docs/hybrid/roadmap.md`).
- `[U]` Client and server are one process on one node. No network, no serialization, no key
  custody — the boundary is a class boundary, not a service boundary.
- `[U]` Encrypted **token-index lookup** (the tokenizer/embedding table) is out of scope; the
  encrypted lineage starts at already-embedded numeric vectors.
- `[U]` `[V]` 663 s is a single sample on a shared-partition node. `host_load1` reaches ~30
  during the score stage, some of it this process's own threads. Per `CLAUDE.md`, latency
  claims need a dedicated or otherwise measured-clean host and more than one sample.
- `[U]` The mask plaintexts (`head_mask_plain`, `score_isolation_plain`, `lane_mask_plain`,
  `copy_active_plain`, `repeated_plain`) are still encoded fresh on every use — ~17.4 k CPU
  encodes concentrated in exactly the stage where GPU utilization is lowest. Untouched by
  T123.
- `[A]` The per-plaintext memory arithmetic in §7 is derived from limb count × ring size, not
  instrumented; the 48.6 GiB total is `[V]` measured.

---

## Appendix — the constants, in one place

| Constant | Value | Why |
|---|---|---|
| `D` | 768 | DNAGPT-0.1b embedding width |
| `T` | 103 | tokenized GSR prompt length (the full prompt) |
| `HEADS` / `HEAD_DIM` | 12 / 64 | 12 × 64 = 768 |
| `MLP_DIM` | 3072 | 4 × D |
| `PACK_WIDTH` | 1024 | 768 padded to a power of two for BSGS |
| `COPIES` | 4 | MLP_DIM / D |
| `TOKEN_BATCH` | 8 | tokens per ciphertext (innermost) |
| `SLOTS` | 32768 | 4 × 1024 × 8 = RING_DIM / 2 |
| `TOKEN_GROUPS` | 13 | ⌈103 / 8⌉ |
| `BSGS_N1` / `BSGS_N2` | 32 / 32 | 32 × 32 = PACK_WIDTH |
| `RING_DIM` | 65536 | polynomial degree N |
| `MULT_DEPTH` | 13 | levels of multiplication budget |
| `SCALE_BITS` / `FIRST_MOD_BITS` | 50 / 60 | fixed-point precision per level |
| `LARGE_DIGITS` | 3 | HYBRID key-switching digits |
| `EPS` | 1e-5 | LayerNorm epsilon (matches upstream) |
| `TOL` | 4e-2 | gate tolerance; `[V]` achieved 4.6e-9 |
