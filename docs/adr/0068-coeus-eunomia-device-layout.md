# ADR 0068: Bind Coeus scalar storage to Eunomia device layout

Status: Accepted (retroactive)

Date: 2026-09-03

Board item: `COEUS-SEMVER-BUDGET-IDENTITY-2026-09-03`

## Context

Hephaestus `ComputeDevice::Buffer<T>` and its provider operation traits require
Eunomia's `Pod`. Coeus `Scalar` previously required only `bytemuck::Pod`, so
the generic Hephaestus bridge accepted a host-layout type that could not be
proven valid for the provider buffer. The failure appeared at every provider
family boundary when Coeus was resolved against Hephaestus revision `076fe32`:
elementwise, reductions, scans, convolution, matmul, pooling, random
initialization, rotate-half, and sliding-window dispatch.

The two marker traits describe related but distinct contracts. `bytemuck::Pod`
remains the host and existing Coeus byte-layout contract; Eunomia's `Pod` is
the first-party device-layout contract. Keeping both on the scalar seam makes
the cross-device invariant explicit without a conversion wrapper or a
backend-specific implementation.

## Decision

Extend the sealed `coeus_core::Scalar` trait with `eunomia::Pod`. All shipped
scalar implementations already satisfy the Eunomia marker, so the change
preserves the available scalar set while making device-buffer validity a
compile-time property of the canonical numeric seam.

Use Eunomia's native layout functions in Hephaestus test buffers. Do not
reinterpret Eunomia layout values through a second marker vocabulary in the
test double. Public Hephaestus dispatch traits and storage declarations carry
the Eunomia bound where their generic parameters are used directly in
Hephaestus provider types; `Scalar` carries the same bound for operation
families that receive Coeus scalars.

## Alternatives rejected

- Keep `bytemuck::Pod` as the only Coeus bound: rejected because it leaves the
  provider boundary unable to prove the buffer contract and reproduces the
  compile failure for every generic operation family.
- Add a downstream adapter from bytemuck to Eunomia: rejected because the
  marker traits are unsafe layout contracts, not runtime-convertible values;
  an adapter would hide the missing proof and duplicate the provider seam.
- Change Hephaestus back to bytemuck: rejected because Eunomia owns the Atlas
  numeric and device-layout vocabulary and the provider contract already uses
  it.

## Invariants

- Every `Scalar` value used by the Hephaestus bridge satisfies both the host
  byte-layout and first-party device-layout contracts.
- Provider dispatch remains generic and monomorphized; no trait object,
  conversion buffer, host round trip, or fallback path is introduced.
- Device COW detachment remains an on-device copy and does not add a download.
- Test buffers use the same Eunomia byte-layout operations as the production
  contract and assert copied values, not only successful calls.

## Consequences

The Coeus-Hephaestus crate gains a direct Eunomia dependency because its public
generic bounds name the provider-owned marker. Existing direct `cutile-rs`
requirements move to the available `0.3.1` package set so the CUDA feature
graph resolves as one version family. The local CUDA build requires the
installed CUDA 13.3 toolkit and a DLL search path with Windows system DLLs and
MSYS2 UCRT before Codex/miniforge native-DLL directories; this is environment
setup, not a repository fallback.

## Verification

- Coeus core semver check against `origin/main`: 196 checks passed, 58 skipped;
  no semver update required.
- Hephaestus, WGPU, and CUDA all-target checks and warning-denied Clippy pass.
- Hephaestus nextest passes 7/7; WGPU nextest passes 142/142; CUDA nextest
  passes 118/118.
- Coeus core and WGPU doctests pass 33/33 and 5/5; CUDA doctests pass 2/2.
- The Hephaestus semver baseline cannot be built against the current provider
  revision because the historical Coeus source predates the required Eunomia
  `Pod` bound; this is recorded as an upstream-coevolution baseline limitation,
  not accepted as a green semver result.
