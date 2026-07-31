# ADR-0043: CUDA native dispatch boundary

- Status: Accepted
- Date: 2026-07-28
- Change class: `[arch] [major]`

## Context

`CudaBackend` selected native kernels when their launch contracts matched, but
binary, unary, matrix, and reduction misses downloaded device storage, executed
the operation through `SequentialBackend`, and uploaded the result. Builds
without the `cuda` feature also implemented the CPU backend contract under the
name `cuda-cpu`. Backend identity therefore did not determine execution
identity, provider failures were hidden, and device operations incurred host
allocation and transfer costs.

## Decision

CUDA elementwise, matrix, reduction, and fused operations execute only through
the selected native CUDA or Hephaestus provider path. A missing context,
unsupported operation or layout, or rejected launch returns
`CudaBackendError`. Builds without the provider feature retain metadata and
storage types for compilation, but implement no Coeus mathematical backend
traits. Fused entry points return typed unavailability when the provider
feature is disabled.

The workspace dependency disables `hephaestus-cuda` default features. The
Coeus `cuda` feature is the sole manifest owner that enables the provider's
CUDA toolchain, so no-provider builds neither compile nor link CUDA support.

CPU computation remains owned by Coeus-Leto and is selected by using a CPU
backend. The CUDA operation families in this decision do not invoke that path
as recovery.

## Alternatives

- Retain the CPU path and log the backend change. Rejected because a reported
  CUDA operation would still execute with different performance, memory, and
  numerical behavior.
- Add a runtime fallback flag. Rejected because it creates a second public
  execution contract and preserves the ownership defect.
- Panic on unsupported requests. Rejected because provider capability is
  input- and environment-dependent and must remain a typed failure.

## Consequences

Previously accepted CUDA operations without native coverage now return errors.
This is a breaking behavioral correction. It removes host staging allocations
from the affected CUDA paths and makes missing provider coverage observable;
no performance gain is claimed without matched measurements.

Optimizer CPU capability paths remain a separate tracked migration. Convolution
and attention now use provider-owned CPU and accelerator contracts. WGPU/CUDA
reduction and aliased elementwise kernels also require provider-owned
Hephaestus contracts before the dispatch audit closes.

## Verification

- Compile `coeus-cuda` without default features and verify the unavailable
  backend identity and typed fused-operation failures.
- Compile and test the CUDA feature under the MSVC CUDA toolchain.
- Run warning-denied Clippy, doctests, and the exact-head provider matrix.
- Scan `coeus-cuda` mathematical dispatch for `SequentialBackend`,
  `copy_to_host`, and fallback calls.
