# ADR 0037: Assign storage construction to backends

- Status: Accepted
- Date: 2026-07-28
- Revised: 2026-09-28 — remove construction from `Storage`; backends and
  concrete storage types now own allocation policy.
- Scope: Coeus CPU, WGPU, CUDA, ROCm, Metal, and generic Hephaestus storage
- Change class: `[arch]`/`[major]`

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

`Storage` also exposed a static `allocate` factory even though the trait's
remaining role is to describe and mutate an allocation that already exists.
That factory duplicated concrete construction policy in `CpuStorage`,
`CowStorage`, and `HephaestusStorage`. It also made a transparent COW wrapper
responsible for creating its inner storage rather than wrapping an existing
owner.

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

Remove `Storage::allocate`. Storage traits describe existing allocations only.
`CpuStorage::new` owns zero-initialized CPU construction, and the sequential
and Moirai `ComputeBackend::allocate` implementations call it directly.
`HephaestusStorage` retains its inherent `new` and `uninitialized`
constructors, where provider placement and initialization policy are known.
`CowStorage::new` continues to wrap an existing storage owner and has no
generic construction path. `ComputeBackend::allocate` remains the generic
backend construction boundary, so Tensor signatures and allocation semantics
do not change.

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
- Retain a static factory on `Storage`: rejected because allocation policy
  belongs to the backend or concrete storage type, while `Storage` only needs
  to expose an existing allocation's capabilities.
- Construct `CowStorage<S>` through `S::allocate`: rejected because it couples
  a representation wrapper to every inner type's construction policy.
- Add provider-specific uninitialized helpers in Coeus: rejected because it
  duplicates the Hephaestus backend seam and forks vendor policy.
- Read from the replacement before the copy: rejected by the provider
  overwrite-before-read contract.

## Verification

The generic Hephaestus regression distinguishes uninitialized and zeroed
allocation paths and verifies exact zero values. The live WGPU regression
verifies both zeroed allocation and clear-after-nonzero values. The existing
Hephaestus backend contracts cover command-stream zero fill for WGPU, CUDA,
ROCm, and Metal. CPU storage tests verify initialization and COW behavior, and
the generic backend-write suite checks fill, zero-fill, and upload values for
all twelve supported scalar types.

This change supplies static allocation-path and value-semantic evidence only;
runtime bandwidth, latency, and resident-memory claims require a controlled
benchmark.

## Revisit trigger

Revisit if a provider cannot supply a real overwrite-before-read allocation or
native zero operation, if a generic storage constructor becomes necessary
outside `ComputeBackend`, or if controlled allocation benchmarks falsify the
expected removal of redundant device writes and host staging.
