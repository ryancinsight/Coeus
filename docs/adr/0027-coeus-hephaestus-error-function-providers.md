# ADR-0027: Add Hephaestus error-function providers

## Status

Accepted — implementation and exact-head provider CI are complete.

## Context

Coeus and Leto expose f32 `erf` and `erfc` operations. The shared Hephaestus
expression seam now owns WGPU, CUDA, ROCm, and Metal forms, but Coeus ROCm and
Metal dispatch still rejected both operations. That left the consumer backend
capability below the existing WGPU/CUDA and CPU vocabulary.

## Decision

Add `UnaryOp::Erf` and `UnaryOp::Erfc` to the existing ROCm and Metal f32
activation-capable dispatch table. ROCm consumes the Hephaestus HIP expressions
and Metal consumes the Hephaestus WGPU-backed expressions through the existing
provider API. Keep integer backends on their arithmetic-only capability
boundary and preserve the typed unsupported-operation error for those types.

Extend both backend integration suites with the existing Leto CPU oracle over
the bounded real-valued input set. The test uses the established f32 tolerance;
it covers native device output semantics rather than merely checking dispatch
success.

## Alternatives rejected

- Keep `erf` and `erfc` unsupported: rejected because the shared Hephaestus
  expressions and CPU contract already exist.
- Reimplement the approximation in Coeus: rejected because dialect syntax and
  kernel ownership belong to Hephaestus.
- Fall back to CPU execution: rejected because it would hide a missing device
  capability and change output placement semantics.

## Verification

Local compilation and nextest pass for the ROCm and Metal packages with the
Hephaestus error-function branch and merged Leto comparison-marker revision
overlaid temporarily. Coeus run `30282267102` passed CUDA job `90031346303`,
Metal job `90031346354`, ROCm job `90031346411`, and WGPU job `90031346421`.
Required-device ROCm job `90031346992` skipped; physical-device execution is
reported separately from adapterless/provider compilation.

## Residual scope

Exact GELU, additional activations, parameterized activations, `lgamma`,
tail-stable `erfc`, and f64/reduced/vector contracts remain separate parity
items. This decision does not claim complete Coeus CPU/backend parity.
