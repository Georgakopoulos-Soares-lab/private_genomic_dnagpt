# Twelve-block public range-control simulation

This directory provides a plaintext-first circuit schedule for the real
DNAGPT `0.1b` T=2 graph. It keeps the released weights and exact model
semantics, replacing only CKKS-incompatible nonlinear functions with fixed
public Chebyshev polynomials. It is a preflight simulation, not an FHE or GPU
measurement.

The schedule addresses four failures found by the original block-0 circuit:

- **A — fixed public domains.** Every block has a domain calibrated from the
  one immutable public fixture with its predeclared `1.25x` margin. This is
  `[A]`, not evidence that the domains generalize. Domains are fixed before a
  simulated query; a value outside them fails closed and is never clipped.
- **B — stable T=2 attention.** The second causal query uses the exact identity
  `p1 = sigmoid(score_11 - score_10)` and `p0 = 1 - p1`. The encrypted circuit
  therefore never materializes the observed denominator as large as
  `16891.85`. A fixed public `[-0.25, 0.25]` zero-crossing guard is included in
  every calibrated delta domain; it is never chosen from an evaluated query.
- **C — public-scaled LayerNorm.** A public power-of-two scale applies
  `inv_sqrt(v) = inv_sqrt(v/s) / sqrt(s)`, putting each inverse-square-root
  polynomial on a normalized interval. The scale comes only from the fixed
  public domain.
- **D — tail-aware GELU.** A degree-127 full-domain polynomial retains the
  released tanh-GELU semantics over each public domain, including the observed
  `[-13.9222, 38.3430]` tails. It does not silently clip values or require an
  encrypted comparison.

Degrees are selected per block from a fixed public candidate set. Each is the
smallest candidate with isolated error at most `1e-4`, subject to a
performance-motivated degree-`127` cap, after which the complete cumulative
graph must still pass the project gate. GELU in blocks 3 and 4 reaches the cap
with isolated max-absolute errors `6.87e-3` and `1.73e-4`; both are retained
because the cumulative dual gate passes with ample margin. This reduces many
early blocks to degree `7` LayerNorm and degree `9`–`39` activation
polynomials. The final head uses degrees `47` (final inverse square root), `95`
(SiLU), and `13` (head inverse square root). Reported balanced multiplication
depths are only polynomial depth proxies; they exclude CKKS rescaling,
plaintext multiplication, rotations, relinearization, and bootstrapping.

The resulting candidate encrypted schedule is one block per refresh epoch:
LayerNorm → QKV → T=2 sigmoid attention → projection/residual → LayerNorm →
MLP/GELU → projection/residual → bootstrap before the next block. This
bootstrap placement is `[A]` until ciphertext level tracing verifies it; the
plaintext simulation establishes correctness of the approximations, not level
capacity or latency.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m fhe.range_control.simulate
```

Without `--tag`, the command prints the result and writes nothing. To create a
new immutable diagnostic under `fhe/range_control/results/`:

```bash
python -m fhe.range_control.simulate --tag public_fixture_v1
```

The writer uses exclusive creation and refuses to overwrite an existing tag.
It writes `..._PASS.json` only when all 12 block gates, every classifier-head
gate, finiteness, and every public-domain check pass. A failed run writes an
explicit `..._FAIL.json` only when a new tag was requested.

Acceptance is the project contract after every block and classifier stage:
global and worst-token relative-infinity error at most `4e-2`, all values
finite, and zero fixed-domain violations.

## Tests

```bash
source .venv/bin/activate
python -m unittest fhe.range_control.test_range_control
```

The small unit tests cover the stable T=2 identity, public scaling, fail-closed
domain checks, wide-domain GELU, and the dual gate. When the ignored generated
fixture is present, an integration test also runs the full 12-block plus
classifier simulation.

## Scope boundary

`T=2` proves the complete released transformer and classifier graph can be
scheduled without the known range explosions. It is not task-representative
GSR inference: the public two-token prefix predicts `A`, while the complete
103-token public prompt predicts `N`. Broader public calibration and an
encrypted run are still required before claiming private task inference.
