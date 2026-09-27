# ADR 0077: Provider-owned cross-product bridge

- Status: Accepted
- Date: 2026-09-27
- Board item: `COEUS-HEPHAESTUS-DOT-NORM-PROVIDER-BINDING` (cross scope)

## Context

`coeus_ops::cross` (`crates/coeus-ops/src/reduction/linalg.rs`) always
materializes both operands to a contiguous host `Vec` via
`BackendOps::copy_to_host` and folds the three cross-product components on
the CPU, regardless of backend. `dot` shared this shape and was rewritten to
compose `sum(a * b)` over the existing on-device `mul`/`sum` seam (PR #439);
`cross` cannot use that composition because its output is itself a vector
assembled from cross terms of the two operands, not a scalar reduction, so no
existing elementwise/reduction op expresses it.

`hephaestus-core` now defines a device-neutral seam for exactly this
operation: `CrossProductOps<D, T>` (hephaestus ADR 0064, PR #343, merged and
past the pinned revision as of PR #439's lockfile advance), implemented by
`hephaestus-host` (CPU) and `hephaestus-wgpu`. Its contract is narrower than
`coeus_ops::cross`'s: `cross_into(&self, device, a, b, out)` operates on flat,
contiguous buffers holding `n` consecutive `(x, y, z)` triples (length `3n`),
with `out[3i..3i+3] = cross(a[3i..3i+3], b[3i..3i+3])`. It carries no
`Layout`/`StridedView` parameter — unlike `MatmulBackend`, `RotateHalfProvider`,
or the other bound families in `coeus-hephaestus`, all of which dispatch over
arbitrary rank via `StridedView`/`ranked::<N>`.

`coeus_ops::cross` supports an arbitrary `dim` and arbitrary shape: it
decomposes the tensor into `(pre, post)` index space around `dim` with
`stride_k = post`. This degenerates to hephaestus's flat contiguous-triple
layout exactly when `dim` is the tensor's last axis and both operands are
contiguous (`post == 1`, so `stride_pre == 3`, i.e. `base = pre_idx * 3`) —
the common case (`torch.cross(a, b, dim=-1)` on a `[..., 3]` tensor). For any
other `dim` or a non-contiguous operand, the three components interleave
with a stride the seam cannot express, and materializing a permuted
contiguous copy first would spend the same host round-trip this ADR removes,
with no measured need driving that generalization now.

`coeus_ops::BackendOps<T>` is a closed six-trait bundle (`ElementwiseOps`,
`MatmulOps`, `ReductionOps`, `ConvOps`, `PoolOps`, `UnfoldFoldOps`); `dot`
and `cross` are free functions bound only on `BackendOps<T> + Default`, with
no linalg sub-trait, so every backend that exists today already computes
`cross` through the same host-fold path. Backend topology is not uniform:
`CudaBackend` (`coeus-cuda`) and `WgpuBackend` (`coeus-wgpu`) are standalone
structs implementing `BackendOps` directly in their own crates; the generic
`HephaestusBackend<P>` wrapper (`coeus-hephaestus`) is instantiated only for
`RocmProvider` and `MetalProvider`. `hephaestus-core`'s `CrossProductOps` is
implemented for host/CPU and WGPU only — CUDA, ROCm, and Metal have no
per-vendor device-API seam for it yet, unlike `MatmulBackend`'s CUDA/WGPU
pair. Rust's coherence rules forbid a blanket "on-device where available,
else host-fold" default-plus-override on one trait for one concrete type, so
the seam cannot be added as an optional capability layered transparently over
every backend; it is added exactly where `MatmulOps` was (ADR 0066): a new
standalone trait every existing backend root type implements explicitly.

## Decision

- Add `CrossOps<T>: ComputeBackend` as a new standalone trait in
  `coeus-ops::backend_ops::traits`, *not* added to the `BackendOps` bundle
  (adding it there would force every current and future backend to gain an
  implementation just to keep compiling — this capability is additive, not a
  `BackendOps`-composing concern). Signature mirrors `MatmulOps`: `fn
  cross(&self, a: &DeviceBuffer<T>, a_layout: &Layout, b: &DeviceBuffer<T>,
  b_layout: &Layout, dim: usize, out: &mut DeviceBuffer<T>, out_layout:
  &Layout) -> Result<(), Self::Error>`.
- Extract the existing host-fold algorithm (unchanged) into
  `coeus_ops::backend_ops::defaults::cross::cross_host_fold`, a free function
  generic over `B: ComputeBackend`, callable by every explicit impl below.
- `coeus_ops::cross` (the `Tensor`-level free function) changes its bound
  from `B: BackendOps<T> + Default` to `B: CrossOps<T> + Default`, keeping
  its existing `to_contiguous_on` materialization and assertions, and
  delegates the fold to `backend.cross(...)`.
- Explicit `CrossOps<T>` impls, one per backend root type (mirroring how
  `MatmulOps` is implemented per root type, never via a blanket over
  `BackendOps`):
  - `CpuBackend`-marked types (`SequentialBackend`, `MoiraiBackend`): call
    `cross_host_fold` — behavior-identical to today.
  - `HephaestusBackend<P>` for `P: HephaestusProvider` (covers
    `RocmProvider`, `MetalProvider` — no `WgpuProvider`/`CudaProvider`
    instance of this wrapper exists): call `cross_host_fold`, generic over
    `P`. hephaestus has no `CrossProductOps` for ROCm or Metal yet; this is
    recorded here and in the impl's doc comment, not silently degraded —
    the follow-on (binding those two once hephaestus ships their
    `CrossProductOps` impls) is this item's own successor, filed once that
    upstream work lands.
  - `CudaBackend` (`coeus-cuda`): call `cross_host_fold` — same gap, same
    reasoning as ROCm/Metal above.
  - `WgpuBackend` (`coeus-wgpu`): add a `cross_product` family to
    `coeus-hephaestus` following the crate's established per-family module
    shape (`rotate_half/`, `matmul/`) — a `CrossProductBackend<T>` narrow
    device-API seam (`type Device`, `type Operations: CrossProductOps<Device,
    T> + Default`, `cross_device()`, `cross_buffer()`,
    `cross_dispatch_error()`, mirroring `MatmulBackend<T>` exactly) plus a
    shared `cross_product::<B: CrossProductBackend<T>, T>(a, b, out)`
    dispatch function. `WgpuBackend` implements `CrossProductBackend<T>` and
    `coeus_ops::CrossOps<T>`: when `dim == a_layout.ndim() - 1` and
    `a_layout.is_contiguous()`, its `cross` method calls
    `coeus_hephaestus::cross_product::<Self, T>`; otherwise it calls
    `cross_host_fold` — the same function every other backend uses for its
    permanent gap, here used for the layout case this bridge does not cover.
- Differential test: `cross` on a `WgpuBackend` tensor (contiguous, `dim =
  ndim - 1`) matches the CPU host-fold path within the derived
  floating-point tolerance for the same inputs.

## Alternatives rejected

- A blanket `impl<B: ComputeBackend> CrossOps<T> for B` default with a
  specific override for `WgpuBackend`: rejected — not expressible in stable
  Rust. `WgpuBackend: ComputeBackend` holds, so the blanket and the specific
  impl overlap; coherence forbids both existing simultaneously without
  unstable specialization.
- A `CrossProductProvider<T>` marker trait conditionally implemented only by
  `WgpuProvider`, with `HephaestusBackend<P>: CrossOps<T>` gated on `P:
  CrossProductProvider<T>`: rejected. `WgpuBackend` is not a
  `HephaestusBackend<P>` instantiation (it is its own struct), so this would
  give `CrossOps` to no backend that currently calls `cross`'s WGPU path and
  would silently drop `cross` availability for `HephaestusBackend<RocmProvider
  /MetalProvider>` (a hard compile-time capability loss, not an acceptable
  regression) unless every provider was forced to implement the marker
  anyway — at which point it adds nothing over the explicit per-root-type
  impls chosen above.
- Generalize the seam call to arbitrary `dim` by permuting to bring it last
  first: rejected. The permute-then-flatten copy pays the same host
  round-trip cost the bridge exists to remove for the case that matters
  today; revisit only if a consumer needs `cross` on a non-last axis on an
  accelerator backend.
- Extend `CrossProductOps` itself to take a `StridedView` like `MatmulBackend`
  and the elementwise seams: rejected for this increment — that is an
  upstream hephaestus change (new trait shape, new WGPU shader path) out of
  this item's scope; record as a candidate follow-on if the last-axis
  restriction proves limiting.
- Route through `ElementwiseOps` primitives instead (three `mul`+`sub` calls
  per component, matching `dot`'s composition): rejected. `dot`'s
  composition works because `sum(a*b)` is already an existing SSOT
  operation; cross has no equivalent decomposition into existing ops without
  introducing per-component gather/scatter, which is more dispatch surface
  than one `cross_into` call, not less.
- Add `CrossOps` to the `BackendOps` bundle: rejected. It would force every
  present and future `BackendOps` implementor to carry an explicit
  `CrossOps` impl merely to keep compiling, for a capability `dot`/`cross`
  never required before; standalone keeps the bundle's six concerns closed.

## Invariants

- `cross_host_fold`'s body is the unmodified pre-existing host-fold
  algorithm; every backend without an on-device seam computes bit-identical
  results to today.
- Dispatch stays monomorphized; no trait object enters the cross path.
- The WGPU on-device path activates only for a statically-checked layout
  condition (`dim` last, contiguous); every other case explicitly calls the
  same shared host-fold function every other backend uses — never a runtime
  probe-and-degrade.
- No new `BinaryOp`/`UnaryOp` opcode; `CrossProductOps` is called directly.
- No performance claim: no controlled baseline was run for this change; the
  benefit is removing the host round-trip on the covered layout, not a
  measured speedup.
