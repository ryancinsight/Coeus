# ADR-0039: Typed CUDA fused-dispatch failures

- Status: Accepted
- Date: 2026-07-28
- Scope: `coeus-cuda` fused elementwise and fused reduction entry points

## Context

The CUDA fused entry points represented every provider, layout, compilation,
and launch failure as `false`. Their public wrappers then evaluated the same
expression through the CPU implementation. That changed the selected backend,
hid provider failures, and could introduce a device-to-host path on a CUDA
tensor. The boolean also discarded the diagnostic needed to distinguish an
invalid layout from a missing driver or a rejected kernel launch.

## Decision

Return `Result<Tensor<T, CudaBackend>, CudaBackendError>` from the public fused
entry points and `Result<(), CudaBackendError>` from the native dispatch
helpers. `CudaBackendError::Fusion` preserves the operation family and the
provider or validation detail. CUDA fused dispatch either completes on CUDA or
returns that error; it never changes to the CPU evaluator after CUDA has been
selected. The feature-disabled `CudaBackend` remains explicitly CPU-backed and
continues to use the CPU implementation by construction.

Kernel-cache poisoning, NVRTC compilation failure, module/function lookup
failure, layout rejection, count/ABI rejection, transfer failure, and launch
failure all remain on the typed error path. Layout metadata continues to be
uploaded directly to the device; no host round trip is introduced.

## Rejected alternatives

- Preserve the boolean and add logging: logging does not preserve failure
  semantics and still permits silent backend substitution.
- Return `Option`: it loses the provider diagnostic and conflates unavailable
  hardware with invalid input.
- Keep a compatibility wrapper returning a tensor: that retains the silent
  fallback and creates two public contracts for one operation.

## Verification

The no-CUDA contract tests assert the `Result` value before comparing the
CPU-backed values. CUDA feature tests assert successful typed dispatch before
checking CPU differential values. Static residual scans must show no boolean
fused dispatch or CUDA-feature CPU fallback in the entry points. Provider
execution and performance deltas require the CUDA feature gate and a matching
device benchmark; this migration makes no runtime or memory claim.
