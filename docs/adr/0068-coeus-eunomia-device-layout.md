# ADR 0068: Bind Coeus scalar storage to Eunomia device layout

Status: Accepted

Date: 2026-09-03 (recorded retroactively as built)

- Revised 2026-11-24: `Scalar` drops its `bytemuck::Pod` supertrait, leaving Eunomia's `Pod` as its only layout marker, because a second marker stated one contract twice and every host byte view has an `eunomia::layout` form; evidence Coeus#695, item `COEUS-BYTEMUCK-RESIDUE`, and `git grep -n bytemuck` matching only Cargo.lock outside Markdown.

Board item: `COEUS-SEMVER-BUDGET-IDENTITY-2026-09-03`

## Context

Hephaestus `ComputeDevice::Buffer<T>` and its provider operation traits require
Eunomia's `Pod`. Coeus `Scalar` previously required only `bytemuck::Pod`, so
the generic Hephaestus bridge accepted a host-layout type that could not be
proven valid for the provider buffer. The failure appeared at every provider
family boundary when Coeus was resolved against Hephaestus revision `076fe32`:
elementwise, reductions, scans, convolution, matmul, pooling, random
initialization, rotate-half, and sliding-window dispatch.

Eunomia's `Pod` and `layout` module own the stack's byte-layout vocabulary
(`bytes_of`, `cast_slice`, `pod_read_unaligned`), so the host byte views Coeus
needs are expressible through the same marker the provider boundary requires.
A second marker adds no information: every shipped scalar satisfies both, and
the only host use of `bytemuck` was `Scalar::has_zero_bit_pattern` plus the
checkpoint payload casts.

## Decision

Extend the sealed `coeus_core::Scalar` trait with `eunomia::Pod` as its only
layout marker. All shipped scalar implementations already satisfy the Eunomia
marker, so the change preserves the available scalar set while making
device-buffer validity a compile-time property of the canonical numeric seam.
Host byte views (`Scalar::has_zero_bit_pattern`, checkpoint encode and decode)
use `eunomia::layout`. `Scalar` is sealed: its supertraits `NumericElement`
(Eunomia) and `CpuUnaryDispatch` (coeus-core) each require a `Sealed` trait
declared in a crate-private `private` module, which no downstream crate can
name, so no downstream crate implements `Scalar`. The only source break is
for code that relied on `T: Scalar` implying `bytemuck::Pod`.

Migration for that code, carried by the next breaking 0.x release:

- Add `T: bytemuck::Pod` beside `T: Scalar` where `T` reaches a
  `bytemuck::Pod`-bounded API; every shipped scalar still implements it.
- Take byte views through `eunomia::layout` (`bytes_of`, `cast_slice`,
  `pod_read_unaligned`) with a direct `eunomia` dependency compatible with
  0.12.1; `coeus_core` re-exports neither `eunomia::Pod` nor
  `eunomia::layout`.
- `bytemuck::pod_collect_to_vec` has no Eunomia counterpart; decode with
  `chunks_exact(size_of::<T>())` mapped through `pod_read_unaligned`, as
  checkpoint decode does.

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
- Retain `bytemuck::Pod` as a second `Scalar` supertrait: rejected because
  Eunomia owns the datatype law, `bytes_of` and `cast_slice` have
  `eunomia::layout` forms, and checkpoint decode replaces
  `pod_collect_to_vec` with per-element `pod_read_unaligned`; two markers
  state one contract twice.
- Change Hephaestus back to bytemuck: rejected because Eunomia owns the Atlas
  numeric and device-layout vocabulary and the provider contract already uses
  it.

## Invariants

- Every `Scalar` value used by the Hephaestus bridge satisfies Eunomia's
  `Pod`, the one marker for host byte views and first-party device layout.
- Provider dispatch remains generic and monomorphized; no trait object,
  conversion buffer, host round trip, or fallback path is introduced.
- Device COW detachment remains an on-device copy and does not add a download.
- Test buffers use the same Eunomia byte-layout operations as the production
  contract and assert copied values, not only successful calls.

## Consequences

The Coeus-Hephaestus crate gains a direct Eunomia dependency because its public
generic bounds name the provider-owned marker. Existing direct `cutile-rs`
requirements move to the available `0.3.1` package set so the CUDA feature
graph resolves as one version family. Hephaestus PR #270 now carries the
merged-Leto/Moirai source-identity correction at `7ca992d`, and Coeus pins that
revision until the provider PR lands. The local CUDA build requires the
installed CUDA 13.3 toolkit and a DLL search path with Windows system DLLs and
MSYS2 UCRT before Codex/miniforge native-DLL directories; this is environment
setup, not a repository fallback.

## Verification

- Coeus core semver check against `origin/main`: 196 checks passed, 58 skipped;
  no semver update required.
- Hephaestus PR #270 includes the merged-Leto/Moirai source correction at
  `7ca992d`; its provider graph has one Leto, Moirai, and Mnemosyne source
  identity.
- Hephaestus, WGPU, and CUDA all-target checks and warning-denied Clippy pass.
- Hephaestus nextest passes 7/7; WGPU nextest passes 142/142; CUDA nextest
  passes 118/118.
- Coeus core and WGPU doctests pass 33/33 and 5/5; CUDA doctests pass 2/2.
- The standalone Coeus workspace gates pass against the 42-source lock:
  check, warning-denied all-target Clippy, nextest 1,359/1,359 with 8 skipped,
  and no-default-feature doctests 160/160 with 2 ignored; warning-denied
  rustdoc, format, lock freshness, and diff hygiene also pass. The default
  doctest invocation still triggers Cargo's known Atlas overlay lock rewrite;
  the no-default feature run is the valid standalone documentation evidence.
- The Hephaestus semver baseline cannot be built against the current provider
  revision because the historical Coeus source predates the required Eunomia
  `Pod` bound; this is recorded as an upstream-coevolution baseline limitation,
  not accepted as a green semver result.
