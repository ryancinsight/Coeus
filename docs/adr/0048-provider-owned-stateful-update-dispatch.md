# ADR 0048: Provider-owned stateful update dispatch

- Status: Accepted
- Date: 2026-08-01
- Board item: `COEUS-STATEFUL-UPDATE-PROVIDER-001`

## Context

Coeus currently owns SGD, Adam, AdamW, RMSProp, and AdaGrad mathematics in its
CPU, WGPU, and CUDA backends. CUDA also retains a host fallback. ROCm and Metal
do not implement the optimizer operation seam. This duplicates contracts now
owned by Leto and Hephaestus, makes backend identity diverge from execution
identity, and prevents validation or dispatch failures from reaching callers.

## Decision

CPU optimizer updates call Leto's scalar-preserving borrowed-view stateful
update APIs directly. WGPU, CUDA, ROCm, and Metal project their existing device
buffers into one generic Coeus-Hephaestus stateful-update bridge. Rule markers
select SGD, Adam, AdamW, RMSProp, or AdaGrad once at the operation boundary, so
the complete provider kernel monomorphizes without per-element dispatch.

`OptimizerOps` returns the backend-associated typed error. The public
`Optimizer::step` contract and scheduler step become fallible, and Rust and
PyO3 callers propagate or map those failures. Parameter construction and all
layout/storage validation complete before provider dispatch; rejected updates
do not mutate parameters or persistent state.

Accelerator stateful updates remain the provider's native `f32` contract.
Coeus does not widen, narrow, download, or route unsupported scalar contracts
through CPU execution. CPU scalar genericity remains native through Leto.

Coeus-owned CPU formulas, WGPU/CUDA optimizer kernels, launchers, and CUDA host
fallbacks are deleted after all callers use the provider seams. The workspace
lock advances to the merged Leto and Hephaestus provider revisions; manifests
retain their canonical git plus version requirements.

## Rejected alternatives

- Retain Coeus kernels behind adapters. This preserves duplicate mathematics
  and a consumer-owned backend dimension.
- Keep infallible optimizer APIs and panic, ignore, or log provider failures.
  This permits false success and partially observed training state.
- Download accelerator storage for Leto execution. This violates backend
  identity, zero-copy ownership, and failure atomicity.
- Add generic accelerator scalar conversions around Hephaestus's `f32`
  contract. This is fake generic execution and changes numerical semantics.

## Consequences

This is a breaking public contract change because optimizer and scheduler
steps return `Result`. All in-repository callers migrate in the same change;
no compatibility entry point remains. Behavioral evidence consists of direct
Leto CPU contracts, shared Hephaestus differential contracts, backend-specific
device tests, and preflight failure-atomic negative cases. Runtime, memory, and
binary-size improvements require separate controlled measurements.
