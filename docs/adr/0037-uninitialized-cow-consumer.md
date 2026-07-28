# ADR 0037: Use overwrite-before-read allocation for accelerator COW

- Status: proposed
- Date: 2026-07-28
- Scope: Coeus WGPU, CUDA, and generic Hephaestus storage uniqueness
- Change class: `[arch]`/`[minor]`

## Context

Coeus storage uniqueness detaches shared accelerator storage by allocating a
replacement in the source buffer's memory tier and copying the complete source
buffer on-device. The copy writes every element before the detached storage is
exposed. Requesting zero-initialized storage for that replacement therefore
adds an initialization pass that the copy immediately overwrites on CUDA and
ROCm.

## Decision

Keep `new` and other ordinary storage construction on
`alloc_zeroed_with_hint`. Use `alloc_uninitialized_with_hint` only for COW
replacement buffers, followed immediately by the existing synchronous
`ComputeDevice::copy_buffer` contract. The generic Hephaestus test device
implements the seam with the same value-semantic storage model.

The provider owns the allocation behavior: Coeus does not call CUDA, HIP,
WGPU, or Metal APIs directly and does not add a host staging fallback. The
source memory tier remains the replacement tier.

## Alternatives rejected

- Continue zeroing every COW replacement: rejected because CUDA and ROCm
  perform a redundant full-buffer write before the full device copy.
- Add provider-specific uninitialized helpers in Coeus: rejected because it
  duplicates the Hephaestus backend seam and forks vendor policy.
- Read from the replacement before the copy: rejected by the provider
  overwrite-before-read contract.

## Verification

The generic Hephaestus storage regression continues to assert copied values,
retained values, tier preservation, one device copy, and zero COW downloads.
Coeus WGPU and generic Hephaestus all-target checks plus the CUDA feature
all-target check pass against the provider branch; the focused generic storage
Nextest test passes. The Hephaestus provider seam merged in PR #136 at
`da785b53`; exact-head provider/consumer CI remains required for this consumer.
This change supplies static allocation-path evidence only; runtime bandwidth,
latency, and resident-memory claims require
a controlled benchmark.

## Revisit trigger

Revisit if a provider cannot provide a real overwrite-before-read allocation or
if a controlled COW benchmark shows its initialization pass is not material.
