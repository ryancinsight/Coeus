# ADR-0041: CUDA pooling index contract

- Status: Accepted
- Date: 2026-07-28
- Scope: `coeus-cuda` pooling launch validation

## Context

CUDA pooling validated each layout field and launch parameter independently
against the unsigned 32-bit ABI. The kernels then formed physical offsets as
`offset + sum(index * stride)` and formed window coordinates with signed
32-bit arithmetic. Individually representable fields did not prove either
derived expression representable. A strided layout could therefore address
beyond its allocation or wrap a physical index, while large pooling
dimensions or parameters could overflow the kernel's signed coordinate
expressions.

The physical layout/storage proof also existed separately in fused and
unfold/fold dispatch. Adding a third operation-local copy would create
divergent memory-safety contracts.

## Decision

Own physical CUDA layout/storage validation in
`kernels::validation::layout_fits_cuda_storage`. The helper proves the maximum
physical offset is representable by the CUDA layout ABI, remains inside the
device allocation, and rejects writable zero-stride aliasing. Fusion,
unfold/fold, and pooling consume this single contract.

Pooling additionally validates the complete forward and backward coordinate
extrema before kernel compilation. For every spatial axis, the last forward
window coordinate and the largest backward input-plus-padding term must fit
the signed 32-bit domain used by the CUDA source. The validation is
allocation-free and executes once at the operation boundary; kernel loops and
monomorphized scalar dispatch remain unchanged.

## Rejected alternatives

- Widen only the CUDA source to 64-bit integers: physical layout indices still
  use the published 32-bit ABI, and allocation bounds would remain unchecked.
- Clamp or wrap oversized coordinates: either changes pooling semantics and
  can redirect an access to unrelated device memory.
- Copy nonconforming layouts to contiguous buffers: that hides the selected
  layout contract and introduces an implicit allocation and transfer path.
- Keep operation-local storage checks: fusion, unfold/fold, and pooling would
  maintain parallel definitions of one device-memory invariant.

## Verification

Pure validation tests cover exact strided storage capacity, undersized
allocation rejection, physical-offset overflow, writable alias rejection,
valid forward/backward extrema, and signed-coordinate overflow. The
feature-enabled package check and warning-denied package Clippy establish the
compiled contract. Exact-head run `30391721824` passes CUDA
(`90384681039`), WGPU (`90384681127`), Metal (`90384681124`), and ROCm
(`90384681137`); required-device ROCm (`90384681768`) is intentionally
skipped. Hosted CUDA execution supplies the device evidence because the local
Windows GNU linker cannot resolve `-lcuda`.
No runtime, bandwidth, or resident-memory improvement is claimed without a
controlled measurement.
