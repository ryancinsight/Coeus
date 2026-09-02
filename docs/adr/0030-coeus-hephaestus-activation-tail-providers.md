# ADR 0030: Route Coeus activation-tail operations through Hephaestus

## Status

Accepted

Implementation note: implementation is complete. Locked metadata, focused non-CUDA
nextest, warning-denied Clippy, workspace doctests, warning-denied rustdoc, and
the MSVC CUDA feature compile check pass. Focused CUDA nextest passes 6/6 with
real device execution; exact-head provider CI remains pending.

## Context

Coeus and Leto already expose f32 `Mish`, `MishGrad`, `Elu`, and `EluGrad`
operations. Before this cutover, Coeus WGPU and CUDA carried local elementwise
expressions while the ROCm and Metal provider dispatch tables stopped at the
earlier activation set. Hephaestus PR #123 added the shared `MishOp`,
`MishGradOp`, `EluOp`, and `EluGradOp` expression markers and exported them
through all four vendor crates. The consumer must use those provider-owned
expressions instead of duplicating dialect source or retaining a compatibility
route.

## Decision

Route the four operations through the existing provider seams:

- WGPU uses the contiguous and rank-bounded strided Hephaestus unary APIs.
- CUDA uses the contiguous and dynamic-rank strided Hephaestus unary APIs.
- ROCm and Metal use their existing f32 activation provider dispatch macros.

The provider expressions define

`Mish(x) = x * tanh(ln(1 + exp(x)))`,

`Mish'(x) = tanh(softplus(x)) + x * (1 - tanh(softplus(x))²) * sigmoid(x)`,

`ELU(x) = x` for `x >= 0`, otherwise `exp(x) - 1`, and

`ELU'(x) = 1` for `x >= 0`, otherwise `exp(x)`.

Backend tests compare each provider result with the Leto CPU oracle on signed
f32 inputs. Integer provider requests stay on typed unsupported paths. An
activation-tail layout or aliasing request outside the provider contract
returns the provider or backend error directly; it never enters a Coeus-local
shader, PTX, or CPU compatibility path.

## Alternatives rejected

- Keep the operations on Coeus-local WGPU/CUDA expressions: rejected because
  Hephaestus owns the dialect-specific expression vocabulary.
- Add ROCm/Metal local kernels: rejected because it duplicates provider-owned
  operation logic.
- Fall back to Leto CPU when a provider path is unavailable: rejected because
  it masks accelerator capability failures and changes backend semantics.
- Add an adapter or compatibility re-export: rejected because the merged
  Hephaestus markers are the canonical provider contract.

## Verification

The focused non-CUDA nextest selection passed 307/307 tests, including the CPU
differential cases and WGPU, ROCm, and Metal forward/gradient parity. Locked
metadata, warning-denied Clippy, workspace doctests (153 passed, 2 ignored),
and warning-denied rustdoc pass; the CUDA-feature and no-default-feature
compile checks also pass. Focused MSVC CUDA nextest passes 6/6 tests, and a
focused CPU/WGPU/ROCm/Metal activation-tail lane passes 10/10. Both lanes
include transposed-layout provider execution. Hosted exact-head provider and
consumer CI remains a separate evidence tier.

## Residual scope

This decision covers the four unparameterized f32 activation-tail operations.
It does not add parameterized activation variants, reduced/vector precision
contracts, or a release/version transition.
