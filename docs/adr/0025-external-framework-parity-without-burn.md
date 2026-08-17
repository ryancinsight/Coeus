# ADR-0025: External-framework parity evidence without Burn

## Status

Accepted

Revision note: Ratifies the already-landed removal in `a365b25e`
(`refactor(coeus-nn): remove legacy Burn benchmark`, MS-442) and
`1c460223` (`feat(coeus)!: Complete provider refresh`, which deleted
`burn_live_parity.rs`).

## Context

The project goal was stated as "complete parity as Burn with testing and
benchmarking against Burn". MS-442 removed the Burn dev-dependency, the
`coeus-nn` Burn comparison harness, and the Burn differential test file, on
the rationale that the committed lock graph should not contain Burn and that
NN correctness is covered by native analytical and provider-conformance
tests.

That left a contradiction: the stated goal named Burn explicitly, but no
Burn comparison remained anywhere in the workspace. This ADR resolves it.

## Decision

Burn is not restored. External-framework parity evidence for Coeus is
carried by the Python differential suites against PyTorch, JAX, and MLX.

Measured coverage at this revision (`crates/coeus-python/tests/`):

| Suite                   | Tests |
| ----------------------- | ----: |
| `test_pytorch_parity.py`|   413 |
| `test_jax_parity.py`    |   215 |
| `test_mlx_parity.py`    |    70 |
| **Total**               | **698** |

PyTorch and JAX coverage is broad enough to serve as the primary
external-correctness oracle, which is what makes the Burn comparison
redundant rather than merely inconvenient. Burn's *semantic conventions*
remain referenced where they informed a contract (for example the PReLU
`0.25` default and the interpolate input conventions); those are
documentation references, not dependencies.

Benchmarking is now intra-Coeus: each `nn_bench` group compares
`SequentialBackend` against `MoiraiBackend`, measuring the production layer
code paths rather than an external framework.

## Alternatives rejected

- **Isolated `coeus-burn-parity` crate** (Burn dev-dependency held in a
  workspace member excluded from the shipped lock graph). Preserves both
  goals, but the maintenance cost of a second Burn-pinned build graph is not
  repaid given the breadth of the PyTorch/JAX suites, which already cover the
  same operators against more widely used references.
- **Reverting MS-442.** Restores Burn benchmarking fastest, but re-introduces
  Burn to the committed lock graph — the specific outcome MS-442 set out to
  remove — and discards landed work.

## Consequences

- The "benchmark against Burn" objective is formally retired. Performance
  comparison against Burn is no longer a Definition-of-Done criterion for
  any NN family; `nn_bench` rows are Sequential-vs-Moirai only.
- Adding a new NN module no longer requires checking whether a Burn
  equivalent exists to build a comparison row against. This removes the
  per-family Burn-surface audit that previously gated benchmark rows for
  MaxPool3d/AvgPool3d and Bilinear.
- Burn-specific numerical conventions that were previously machine-checked by
  `burn_live_parity.rs` are now asserted only against PyTorch/JAX. Where a
  Coeus contract was derived from Burn rather than PyTorch, that derivation
  now rests on the doc comment alone.

## Residual risk

- **MLX parity is not executed on the development platform.** The suite
  self-skips (its header records this) and MLX exposes no `f64`, so those 70
  tests are f32-only and contribute no verified evidence on Windows. Treat
  the effective external oracle as PyTorch + JAX (628 tests); MLX coverage is
  aspirational until run on a supporting platform.
- No performance regression signal against any external framework remains.
  A Coeus-wide slowdown that leaves the Sequential/Moirai *ratio* unchanged
  would not be caught by `nn_bench`; only the absolute criterion baselines
  would show it.
