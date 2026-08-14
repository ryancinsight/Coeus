# ADR-0036: Keep accelerator COW copies on-device

## Status

Accepted — the generic Coeus Hephaestus storage increment and native WGPU/CUDA
consumer cutover are implemented.

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

The implementation remains generic over the provider and scalar type. Native
Coeus WGPU and CUDA storage consumers call the same seam rather than
reimplementing provider transfer mechanics. The change does not add vendor
imports to the shared storage contract, host fallback logic, or a second COW
algorithm.

## Alternatives rejected

- Retain the host round trip: rejected because it allocates O(n) host storage
  and performs two avoidable transfers.
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
