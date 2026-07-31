# ADR 0037: Separate accelerator initialization contracts

- Status: Accepted
- Date: 2026-07-28
- Revised: 2026-07-31 — extend overwrite-before-read allocation from COW
  detachment to the compute-backend allocation contract and add explicit
  provider-native zero allocation and zero fill.
- Scope: Coeus CPU, WGPU, CUDA, ROCm, Metal, and generic Hephaestus storage
- Change class: `[arch]`/`[minor]`

## Context

Coeus storage uniqueness detaches shared accelerator storage by allocating a
replacement in the source buffer's memory tier and copying the complete source
buffer on-device. The copy writes every element before the detached storage is
exposed. Requesting zero-initialized storage for that replacement therefore
adds an initialization pass that the copy immediately overwrites on CUDA and
ROCm.

The same distinction applies to `ComputeBackend`. Kernel output allocation is
documented as uninitialized and every caller must overwrite it before reading,
but accelerator implementations returned zeroed storage. Conversely,
`Tensor::zeros_on` requested that zeroed storage and then performed a second
full-buffer zero fill. WGPU and generic ROCm/Metal fill implemented that second
pass by allocating and uploading a destination-sized host vector.

## Decision

Keep storage `new` construction on `alloc_zeroed_with_hint`. Add explicit
`ComputeBackend::allocate_zeroed` and `ComputeBackend::fill_zero` methods with
CPU-compatible defaults. `Tensor::zeros_on` uses `allocate_zeroed` exactly
once. WGPU, CUDA, and generic Hephaestus `allocate` implementations use
`alloc_uninitialized_with_hint`; their `allocate_zeroed` implementations use
the provider's zeroed allocation. ROCm and Metal runtime wrappers preserve
that generic allocation split.

WGPU, CUDA, ROCm, and Metal override `fill_zero` with Hephaestus
command-stream clears. Each concrete accelerator `fill` implementation detects
the all-zero representation once at the operation boundary and routes it
through `fill_zero`. The open `HephaestusProvider` trait remains unchanged, so
external provider implementations retain source compatibility. Arbitrary
nonzero fill remains a separate operation.

COW replacement buffers continue to use `alloc_uninitialized_with_hint`,
followed immediately by the synchronous `ComputeDevice::copy_buffer`
contract. Matmul scratch uses `allocate_zeroed` directly instead of combining
an allocation with a second explicit fill.

The provider owns the allocation behavior: Coeus does not call CUDA, HIP,
WGPU, or Metal APIs directly and does not add a host staging fallback. The
source memory tier remains the replacement tier.

## Alternatives rejected

- Continue zeroing every overwrite-only allocation: rejected because it
  performs a redundant full-buffer write before a kernel or device copy.
- Keep `Tensor::zeros_on` as zeroed allocation followed by zero fill: rejected
  because it duplicates work and creates destination-sized host staging on
  backends whose arbitrary fill path uploads a host slice.
- Add a required method to the open `HephaestusProvider` trait: rejected
  because it would break external provider implementations; concrete Coeus
  runtimes bind the existing Hephaestus command-stream clear instead.
- Add provider-specific uninitialized helpers in Coeus: rejected because it
  duplicates the Hephaestus backend seam and forks vendor policy.
- Read from the replacement before the copy: rejected by the provider
  overwrite-before-read contract.

## Verification

The generic Hephaestus regression distinguishes uninitialized and zeroed
allocation paths and verifies exact zero values. The live WGPU regression
verifies both zeroed allocation and clear-after-nonzero values. The existing
Hephaestus backend contracts cover command-stream zero fill for WGPU, CUDA,
ROCm, and Metal. Exact-head backend CI is required before closure.

This change supplies static allocation-path and value-semantic evidence only;
runtime bandwidth, latency, and resident-memory claims require a controlled
benchmark.

## Revisit trigger

Revisit if a provider cannot supply a real overwrite-before-read allocation or
native zero operation, or if controlled allocation benchmarks falsify the
expected removal of redundant device writes and host staging.
