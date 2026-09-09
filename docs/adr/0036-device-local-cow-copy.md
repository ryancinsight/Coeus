# ADR 0036: Keep accelerator COW copies on-device

## Status

Accepted

Revision 2026-09-08: [backend writes](../backlog.md#coeus-backend-write-ownership)
must detach shared storage at the mutation boundary. Explicit detachment tests
did not cover direct backend fill and upload calls. The subsequent
[kernel-output audit](../backlog.md#coeus-device-output-ownership) finds the
same ownership gap in mathematical dispatch and temporary vendor storage
bridges. This revision extends the ownership boundary to every mutable output.

## Context

`HephaestusStorage::make_unique` detached shared storage by allocating a
full-size host vector, downloading the source buffer, and uploading the host
copy. This made copy-on-write proportional to host memory, discarded the
source allocation tier, and performed two unnecessary transfers.

## Decision

Detach shared storage through the provider-native `ComputeDevice::copy_buffer`
contract. The consumer acquires one device reference, allocates the
replacement with `PlacementHint::Tier(source.tier())`, copies the complete
typed buffer device-to-device, and retains the replacement behind the existing
`Arc` handle. The shared Hephaestus contract requires the copy to complete
before returning, so the storage mutation does not expose an in-flight buffer.

The implementation remains generic over the provider and scalar type. CUDA
and WGPU select `HephaestusStorage<P, T>` as their device storage directly.
The vendor storage structs and `from_arc` conversion path are removed. A
shared output must remain owned by the caller when detachment installs a new
buffer; detaching a temporary bridge would discard the computed result.
Read-only inputs pass by borrow and do not acquire additional shared owners.
The CUDA build without device support uses the existing CPU storage type and
continues to expose no mathematical CUDA implementation.

`ComputeBackend::fill`, `fill_zero`, and `copy_to_device` preserve the values
of every other storage clone. Each provider write detaches its destination
before obtaining the buffer used by the write. Fill paths delegate to the
same upload or clear boundary so no separate copy-on-write algorithm exists.
An exclusive destination needs no copy. A shared destination installs its
replacement only after the device copy completes.

Mathematical dispatch takes mutable storage for each writable output, including
additive gradients and optimizer states. It detaches each writable owner before
projecting the buffer used for execution and preserves the complete allocation
for partial or additive writes. Attention, convolution and optimizer requests
preflight all outputs before detachment. Single-output requests can detach
before provider validation; a rejected request preserves values but need not
preserve allocation identity. Separate cloned owners detach independently;
intrinsic overlap inside a writable layout remains invalid. This decision
does not promise rollback after a device failure during execution.

Logical element counts do not imply GPU transfer alignment. The WGPU provider
owns physical padding and copies or clears the allocated byte extent; Coeus
passes the complete typed buffer through the existing device contract.

## Alternatives rejected

- Retain the host round trip: rejected because it allocates O(n) host storage
  and performs two avoidable transfers.
- Retain vendor wrappers with copy-back assignments after dispatch: rejected
  because every write family would need to synchronize two owners and read
  bridges would introduce artificial sharing.
- Add provider-specific COW implementations: rejected because it duplicates
  storage ownership logic across WGPU, CUDA, ROCm, and Metal consumers.
- Make `StorageMut::make_unique` silently recover from provider failure:
  rejected because the current trait is infallible and a fallback would hide a
  failed device operation. The fallible storage-boundary migration remains a
  separate tracked item.

## Verification

The generic storage contract uses a fake device implementation only as a test
double for the public provider seam. It asserts copied values, source-tier
preservation, exactly one device copy, and zero downloads during COW. Provider
integration compilation and the WGPU, CUDA, ROCm, and Metal contract suites
remain the backend execution evidence. The native WGPU and CUDA storage tests
download both COW results and assert equal values. No runtime performance claim
is made without a matched device benchmark.

Direct backend write cases share one scalar/backend-parameterized oracle:
download the modified destination and its untouched clone, then compare both
against exact small-integer values. The cases exercise fill, zero-fill and
upload for empty, odd-length and aligned allocations on CPU and actual GPU
providers. These cases establish successful-write ownership semantics;
provider failure propagation remains part of the fallible storage migration.

Kernel-output tests instantiate one operation-specific oracle over the shipped
scalar/backend contracts. They compare dense and strided destinations, their
retained clones, and elements outside the logical view. Analytical cases cover
multi-output attention, accumulated convolution and pooling gradients, window
fold replacement, staggered differences, and optimizer state snapshots.
Invalid-request cases assert unchanged values after rejection. These checks
establish output ownership, not recovery from device failure.

## Migration

Use `<CudaBackend as ComputeBackend>::DeviceBuffer<T>` or
`<WgpuBackend as ComputeBackend>::DeviceBuffer<T>` when naming the selected
storage, or `HephaestusStorage<P, T>` for provider-generic code. `CudaStorage`
and `WgpuStorage` no longer exist. Allocate through the backend or adopt a
provider buffer with `HephaestusStorage::from_buffer`; `from_arc` is removed.
CUDA raw-pointer callers use `storage.buffer().raw()`. Allocation-reuse
diagnostics use the generic `allocation_id()` and compare identities only
while both allocations remain live. No public mutable Arc field is exposed.

Callers of shared matmul, pooling, unfold/fold and staggered dispatch supply
`&mut` storage for output arguments. Mathematical backend trait signatures
already take mutable outputs and keep their call shape. Read-only operands
remain shared borrows. No manifest version or release changes accompany
this development migration.
