# Config 2 — GPU sharding (Q/K/V projection split across 2 GPUs)

**What it measures:** the **attention Q/K/V projection step** of one block, T=103,
depth-13, split across **2 GPUs, process-per-GPU**. A CPU-only *writer* serializes
the crypto context + keys + encrypted inputs to a state dir; then two *reader*
processes run **concurrently** — worker A computes the query+key projections
(`--gpu 0`), worker B the value projection (`--gpu 1`), each checked against its
oracle. Q/K/V are independent output ciphertexts, so **no ciphertext crosses
processes** (only context+keys are serialized — which the pinned FIDESlib
supports). Historical measured concurrency benefit on this step: ~1.65×.

> **Scope — read this.** This is the Q/K/V **projection** step only. The readers
> do **not** run attention scores / softmax / GELU / MLP (see the reader source
> header: *"there is no attention/GELU/MLP stage in this reader"*). It is a real,
> accuracy-verified measurement of how much 2-GPU splitting speeds up that step —
> not a full sharded block.
>
> **Why not a full sharded block/e2e?** Continuing past the projection needs Q, K,
> V brought back together for attention = moving a **ciphertext** between the two
> processes. The pinned FIDESlib (786c7600) exposes no ciphertext serialization
> (only context/keys), so process-per-GPU cannot recombine. This is the *same*
> wall that blocks the MLP-merge (Stage-2). A full sharded forward pass would
> need a single-process multi-GPU design (`SetDevices({0,1,2})`), which does not
> exist in this codebase. See `../../README.md` §Known limits.

## Prerequisites
Same as Config 1, plus the `block` fixture must also satisfy the sharding
contract `fhe/gpu_real_scheme_b/fixture_t103_qkvshard.sha256` (same fixture dir).

## Run
Needs **2 of the node's 3 A100s**. From the repo root:
```bash
mkdir -p kimon/logs/config2_sharding
sbatch kimon/configs/config2_sharding/block.sbatch      # edit -A first
```

## Output
`kimon/logs/config2_sharding/qkvshard_<jobid>/`:
- `<writer>.json`, `<reader_querykey>.json`, `<reader_value>.json` — per-worker results.
- `writer.log`, `reader_querykey.log`, `reader_value.log`.
- `state/` — **deleted on exit** (held a serialized secret key; hygiene).

**PASS** = both reader logs contain `REAL_DNAGPT_FIDES_SCHEME_B_SIMD_SHARD_READER_PASS`.
**Speedup** = compare `max(readerA, readerB)` wall-clock vs `readerA + readerB`
(serial-equivalent) from the JSON `encrypted_evaluation_seconds`.

## Notes
- Binaries: `..._simd_shard_writer_t103_depth13`, `..._simd_shard_reader_t103_depth13`.
- 3-way split (query|key|value on 3 GPUs) is possible on TACC's 3-A100 node but
  not wired here; ask if wanted.
- **e2e (sharded 12-block): NOT feasible** with process-per-GPU on the pinned
  FIDESlib (ciphertext-serialization wall, above). Would require an unbuilt
  single-process multi-GPU design. See `../../README.md` §e2e status.
