# ADR 0029: Route Coeus lgamma through Hephaestus providers

## Status

Accepted

Implementation note: implementation and exact-head CI are complete for the f32 forward
provider/consumer boundary.

## Context

Leto already exposes `UnaryOp::Lgamma` as the natural logarithm of the
absolute gamma function. Coeus's WGPU unary kernel rejected the operation,
CUDA had no device expression, and the ROCm and Metal Hephaestus dispatch
tables did not route it. This left the f32 backend capability set below the
CPU contract.

## Decision

Route the operation through the existing provider seams:

- WGPU dispatch uses `hephaestus_wgpu::LgammaOp`, whose WGSL expression uses
  the provider-owned Lanczos approximation, reflection identity, infinity, and
  non-positive integer pole handling.
- Metal dispatch uses the same provider marker and expression through the
  existing Hephaestus Metal/WGPU path.
- CUDA emits the native `lgammaf` device function for both contiguous and
  strided kernels.
- ROCm dispatch uses `hephaestus_rocm::LgammaOp`, whose HIP expression emits
  the native `lgamma` function.

All backend tests compare with the Leto CPU oracle. Finite cases cover
positive and reflected non-integer inputs. Pole cases require matching
positive infinity rather than applying a finite tolerance to `∞ - ∞`.

## Alternatives rejected

- Keep WGPU unsupported: rejected because it violates the existing Leto CPU
  contract.
- Copy the Lanczos formula into Coeus: rejected because Hephaestus owns the
  dialect-specific expression.
- Fall back to Leto CPU: rejected because it hides missing accelerator
  capability and changes the backend execution contract.
- Implement a digamma-based gradient here: rejected because Coeus does not
  currently expose a digamma provider contract; that is a separate operation
  slice.

## Verification

The affected Coeus crates require locked local checks and nextest. Hephaestus
PR #118 passed WGPU `90086428952`, CUDA `90086430178`, ROCm `90086430143`, and
Metal `90086428160`. Coeus PR #231 merged at
`971fab9614b97bd708a716d01684da58fd1331ba`; its consumer jobs passed WGPU
`90088836682`, CUDA `90088836688`, ROCm `90088836731`, and Metal `90088836675`.
Required-device ROCm `90088837591` was skipped because no hosted AMD runner was
dispatched. Hardware execution remains a separate evidence tier and is not
claimed when the required-device job skips.

## Residual scope

This decision covers f32 forward `lgamma` only. It does not claim digamma
gradients, f64/reduced/vector contracts, tail-stable `erfc`, parameterized
activations, or complete parity for non-elementwise Leto operations.
