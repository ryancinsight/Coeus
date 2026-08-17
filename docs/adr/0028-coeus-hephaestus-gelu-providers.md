# ADR-0028: Add Hephaestus exact GELU providers

## Status

Accepted

Implementation note: implementation is complete and exact-head CI is pending.

## Context

Coeus and Leto expose exact f32 `Gelu` and `GeluGrad` operations. Coeus WGPU
and CUDA already contain equivalent expressions, while the ROCm and Metal
Hephaestus dispatch tables reject both operations. The provider-owned
Hephaestus vocabulary must close that consumer capability gap without copying
kernel expressions into Coeus or silently executing on the CPU.

## Decision

Route `UnaryOp::Gelu` and `UnaryOp::GeluGrad` through the existing ROCm and
Metal activation dispatch macros using `hephaestus_rocm::GeluOp`,
`GeluGradOp`, `hephaestus_metal::GeluOp`, and `GeluGradOp`.

The forward contract is

`GELU(x) = 0.5 x (1 + erf(x / sqrt(2)))`.

The gradient contract is

`GELU'(x) = 0.5 (1 + erf(x / sqrt(2))) + x exp(-x² / 2) / sqrt(2π)`.

Both backend suites compare provider output with the Leto CPU oracle over the
existing bounded activation input domain using the established f32 error
bound. Integer dispatch remains on the typed unsupported-operation path.

## Alternatives rejected

- Keep ROCm and Metal unsupported: rejected because exact GELU is already part
  of the CPU and WGPU/CUDA contract.
- Copy WGPU/CUDA expressions into Coeus: rejected because Hephaestus owns
  dialect-specific kernel expressions.
- Substitute `GeluTanh`: rejected because it is a different approximation
  contract and already has its own operation marker.
- Fall back to CPU: rejected because it masks missing accelerator capability.

## Verification

Local ROCm and Metal test-target compilation and nextest pass with the
Hephaestus GELU branch temporarily overlaid. Exact-head WGPU, CUDA, ROCm, and
Metal provider workflows plus the Coeus consumer matrix are required before
this item closes; hardware execution remains a separate evidence tier.

## Residual scope

Parameterized activations, `lgamma`, tail-stable `erfc`, and f64/reduced/vector
contracts remain separate parity items. This decision does not claim complete
Coeus CPU/backend parity.
