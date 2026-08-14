# ADR-0065: Provider-owned batched Frobenius norm

- Status: Accepted
- Date: 2026-08-06
- Board item: `COEUS-AUTOGRAD-HOST-STAGING-RESIDUALS-001` (bounded
  `coeus_ops::frobenius_norm_batched` slice)

## Context

`coeus_ops::frobenius_norm_batched` computed rank-3-and-higher inputs by
copying the complete contiguous tensor to a host `Vec`, folding each matrix in
Rust, and uploading the result. That path made a device-resident operation
CPU-addressable, allocated storage proportional to the input, and bypassed
the provider reduction and elementwise seams.

## Decision

Compose the operation from the existing backend-generic operations:

1. Materialize a contiguous provider tensor only when the input view is
   strided or offset.
2. Square it with the canonical elementwise multiplication operation.
3. Reduce the final two axes with provider-owned `sum_axis` operations.
4. Apply provider-owned `sqrt` and reshape the result to the leading batch
   shape.

The generic operation imports no concrete device implementation. CPU
`BackendOps` implementations dispatch through Leto; WGPU, CUDA, ROCm, and
Metal implementations dispatch through their Hephaestus-backed Coeus seams.
This preserves monomorphized dispatch at the backend operation boundary.

## Invariants

- The rank-2 path keeps its existing scalar-returning contract.
- Rank `>= 3` returns one value per leading batch index with shape
  `input.shape[..rank - 2]`.
- No rank-`>= 3` path calls `copy_to_host`, constructs an input-sized host
  vector, or uses a CPU fallback.
- Strided input materialization remains provider-local and does not mutate the
  source storage.
- The mathematical result remains `sqrt(sum(a * a))` over the final two axes.

## Rejected alternatives

- Retaining the host fold would preserve the provider-residency and memory
  defects.
- Flattening the entire input would lose per-batch output semantics.
- Adding a new fused provider kernel would duplicate a composition already
  expressed by existing SSOT operations without a measured need.

## Verification

- Analytical CPU differential coverage includes contiguous, rank-4, and
  strided inputs; the strided case also checks source preservation.
- `coeus-ops` Nextest passes 209/209, including 7 selected Frobenius tests.
- Locked workspace all-targets check, warning-denied `coeus-ops` Clippy, and
  23 `coeus-ops` doctests pass.
- No runtime or resident-memory improvement claim is made without controlled
  baseline measurements. Hosted provider parity remains the merge gate for
  accelerator instantiations.
