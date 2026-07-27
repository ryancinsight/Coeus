# ADR-0029: Route Coeus lgamma through Hephaestus providers

## Status

Accepted; implementation is complete and exact-head CI is pending.

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

The affected Coeus crates require locked local checks and nextest. Exact-head
WGPU, CUDA, ROCm, and Metal provider workflows plus the Coeus consumer matrix
are required before closure. Hardware execution remains a separate evidence
tier and is not claimed when required-device jobs skip.

## Residual scope

This decision covers f32 forward `lgamma` only. It does not claim digamma
gradients, f64/reduced/vector contracts, tail-stable `erfc`, parameterized
activations, or complete parity for non-elementwise Leto operations.
