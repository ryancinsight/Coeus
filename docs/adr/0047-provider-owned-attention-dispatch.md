# ADR 0047: Provider-owned attention dispatch

- Status: Accepted
- Date: 2026-07-31
- Board item: `COEUS-ATTENTION-PROVIDER-001`

## Context

Coeus currently owns scaled dot-product attention mathematics in its CPU,
WGPU, and CUDA backends. CUDA also contains a host fallback. This duplicates
the CPU contract now owned by Leto and the accelerator contract now owned by
Hephaestus, prevents ROCm and Metal from substituting through the same role,
and makes provider failures impossible to represent at the public operation
boundary.

## Decision

CPU attention calls Leto's borrowed rank-three forward and additive-backward
APIs directly. WGPU, CUDA, ROCm, and Metal bind their existing device buffers
to one generic Coeus-Hephaestus attention bridge, which monomorphizes through
the selected provider's `AttentionOps` implementation. Runtime backend
selection therefore occurs once at the Coeus backend boundary; mathematical
loops do not perform backend or capability checks.

The Coeus-Hephaestus `AttentionBackend` seam owns layout validation, operand
assembly, dispatch, and error mapping defaults. Vendor leaves provide only
device selection, zero-copy buffer projection, and the typed error constructor.
Coeus owns the provider-neutral `f32`/`f64` attention scalar marker; only the
CPU implementation adds Leto's provider-specific scalar bound.

The public attention operation becomes fallible. The `Result` propagates
through autograd, neural-network modules, and PyO3 error mapping. All
in-repository callers migrate in the same change. Coeus-owned attention
kernels, launchers, and host fallbacks are deleted once their callers use the
provider seam.

Grouped rank-two masks remain borrowed. The CPU bridge binds the complete
`[group, key]` view to Leto's `GroupedKeepMask` with a nonzero
batches-per-group contract, then performs one provider operation. Leto
validates every group before output mutation without materializing or copying
mask data.

## Rejected alternatives

- Retain Coeus kernels behind adapters. This preserves duplicate mathematics
  and a consumer-owned backend dimension.
- Keep an infallible API with panic or silent fallback. This hides provider
  validation and execution failures.
- Download accelerator operands into Leto. This violates backend selection,
  zero-copy ownership, and failure semantics.

## Consequences

This is a breaking public contract change because attention forward and
backward return `Result`. Callers use `?` or map the typed backend error.
The migration removes compatibility paths rather than retaining deprecated
wrappers. Behavioral evidence consists of shared Leto/Hephaestus differential
contracts, backend-specific device tests, and preflight-failure-atomic negative
cases; performance claims require separate controlled measurements.
