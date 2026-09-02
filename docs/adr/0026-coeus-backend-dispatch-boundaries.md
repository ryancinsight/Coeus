# ADR 0026: Make unsupported reduction dispatch statically unavailable

- Status: Accepted
- Date: 2026-07-27
- Scope: `coeus-ops` reduction traits and their Coeus/autograd callers

## Context

`ReductionOps` supplied default implementations for `argmax`, `argmin`, and
`topk`. Those defaults copied accelerator storage to host memory, executed a
Coeus/Leto CPU algorithm, and copied the result back. This made an operation
appear available on ROCm, Metal, WGPU, CUDA, and generic Hephaestus backends
without a native provider implementation. It violated backend ownership,
zero-copy expectations, and the provider-first dispatch contract.

CPU backends already implement these operations directly through Leto and
`CpuBackend` exposes the required addressable output seam. The accelerator
providers do not currently expose native selection kernels.

## Decision

The default selection methods are constrained by `Self: CpuBackend`. The public
`coeus_ops::{topk,argmax,argmin}` and autograd `topk` entry points carry the
same bound. CPU calls therefore monomorphize directly to the existing Leto
implementation. Accelerator calls cannot select a host-copy default; they
remain unavailable until their owning provider exposes a native operation.

Native reduction and scan methods remain unchanged. ROCm and Metal continue to
dispatch through Hephaestus, while WGPU and CUDA continue to dispatch through
their native provider paths.

## Rejected alternatives

- Copying accelerator buffers to CPU: preserves the false capability and adds
  allocation, transfer, and synchronization cost.
- Runtime backend-name branching: duplicates capability policy in call sites
  and defers an invariant that the type system can enforce.
- `dyn` operation providers: adds vtable dispatch and erases the existing
  monomorphized backend seam.
- Local accelerator selection kernels: violates upstream provider ownership and
  creates duplicated operation vocabularies.

## Verification

- CPU/Leto selection tests continue to exercise value-semantic results.
- Trait bounds are checked for all workspace backends; non-CPU backends cannot
  satisfy the selection entry points through the default implementation.
- Provider reduction and scan dispatch remains covered by the existing backend
  matrix.

## Follow-up

Native arg-reduction and top-k kernels belong in the owning Hephaestus/provider
operation families. That work is a separate vertical slice and must add
provider conformance and differential tests before widening the bound.
