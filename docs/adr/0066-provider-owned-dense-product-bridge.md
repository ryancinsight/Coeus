# ADR 0066: Provider-owned dense product bridge

Status: Accepted
Date: 2026-08-17
Board item: `ATLAS-COEUS-BACKEND-045`

## Context

`HephaestusBackend<P>` implemented `ElementwiseOps`, `ReductionOps`, and
`ConvOps` but none of `MatmulOps`, `PoolOps`, or `UnfoldFoldOps`. Metal and
ROCm therefore reached the accelerator only for the families the generic layer
already covered, while `CudaBackend` and `WgpuBackend` each carried their own
matmul kernel: a 16x16 shared-memory tiled CUDA C string in
`coeus-cuda/src/kernels/launch_matmul.rs` and a line-by-line WGSL
transliteration of the same algorithm in `coeus-wgpu/src/kernels/matmul.rs`.
One algorithm existed in two dialects with two validation paths, and neither
was reachable from the provider seam that Metal and ROCm use.

`hephaestus-core` already defines the device-neutral seam for this family:
`DenseProductOps<D, T>` (ADR 0044), implemented by all four vendor crates —
`CudaDenseProductOps`, `WgpuDenseProductOps`, `MetalDenseProductOps`,
`RocmDenseProductOps`. The seam was unused by Coeus.

`ComputeBackend` additionally carried `private::Sealed` as a supertrait, with
`private` re-exported publicly from `coeus_core::backend`. The trait's
implementor set spans sibling crates by design — one per vendor — so the seal
was both wrong in principle and inoperative in practice: any downstream crate
could implement the public `Sealed` marker.

## Decision

Add a `matmul` family to `coeus-hephaestus` following the crate's established
convolution shape, and route every backend through it.

- `MatmulProvider<T>` names a provider's `DenseProductOps` bundle.
- `MatmulBackend<T>` is the narrow device-API seam a backend implements:
  device accessor, buffer accessor, and error mapper. Nothing else.
- `matmul::<B, T>` holds the orchestration once — rank-2 layout validation and
  left-padding via `layout::ranked`, `StridedView` assembly, dispatch, error
  mapping — and monomorphizes per backend.
- `HephaestusBackend<P>` gets `MatmulBackend<T>` by blanket impl, so a provider
  declaring `MatmulProvider` gains `MatmulOps` with no further code.

`CudaBackend` and `WgpuBackend` implement `MatmulBackend<T>` and delegate to
the shared function. Their hand-written kernels, launch code, and storage
adapters are deleted, not wrapped. The duplicate public `coeus_wgpu::matmul`
free function is routed through the same seam rather than left as a second
implementation.

Remove the `ComputeBackend` seal: delete the `private` module, the supertrait,
its re-export, and all five `Sealed` impls.

## Alternatives rejected

- Override `batched_matmul` with the seam's `batched_matmul_into`: rejected.
  The inherited default decomposes a rank-3 call into per-slice rank-2 device
  dispatches and supports batch broadcasting (`lhs_batch == 1`); the seam
  method's broadcasting contract is unstated, and no GPU device on the
  development host could validate the difference. Preserving the default keeps
  CUDA and WGPU batched semantics bit-identical to what shipped.
- Migrate `CudaBackend`/`WgpuBackend` wholesale onto `HephaestusBackend<P>`:
  rejected for this increment. It requires `PoolOps` and `UnfoldFoldOps`
  seams that do not exist in `hephaestus-core`, so it cannot be done without
  a half-migrated op family.
- Retain the CUDA tiled kernel behind a capability check: rejected as a dual
  path. The provider seam owns the kernel; a second one in the consumer is the
  fork this ADR removes.
- Seal `ComputeBackend` properly with a crate-private marker: rejected because
  the per-vendor cross-crate impls are the intended extension mechanism.

## Invariants

- Dispatch stays monomorphized; no trait object enters the matmul path.
- `MatmulBackend` carries no operation methods — only the device-API surface.
- Layout validation happens once, in the shared dispatch, for every backend.
- Provider failure remains a typed error; no host fallback is introduced.
- No performance claim is made: the CUDA and WGPU kernels change from the
  consumer's tiled implementation to the provider's, and no controlled
  baseline was run.

## Consequences

`HephaestusBackend<MetalProvider>` and `HephaestusBackend<RocmProvider>` now
satisfy `MatmulOps<f32>`. They do not yet satisfy `BackendOps<f32>`: that
marker requires all six sub-traits, and `PoolOps` and `UnfoldFoldOps` remain
unavailable to the generic layer because `hephaestus-core` has no pooling or
sliding-window device seam. Closing that gap is upstream work in Hephaestus —
a `PoolOps<D, T>` and an `UnfoldFoldOps<D, T>` trait with four vendor impls —
and is tracked separately. Until then the CUDA and WGPU pool and unfold/fold
kernels stay where they are; they are the remaining fork.

Removing the seal widens `ComputeBackend`'s public contract: downstream crates
may now implement it. This is the intended seam behaviour, and is a
`[minor]` surface change.

## Verification

- `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`,
  `cargo nextest run --workspace`, and `cargo test --doc` in Coeus.
- A compile-time bound assertion in `coeus-metal` pins the acceptance
  condition that the provider seam, not a vendor kernel, supplies matmul.
- `cargo semver-checks` for the `coeus-core` and `coeus-wgpu` surface changes.
- Matmul parity suites (`coeus-cuda/tests/cuda/parity/matmul.rs`,
  `coeus-wgpu/tests/.../parity/matmul.rs`) are unmodified. They did not
  execute on the development host: CUDA parity self-skips with no device, and
  WGPU fails at adapter acquisition. Device-side equivalence of the provider
  kernel against the deleted consumer kernel is therefore unverified here and
  must be confirmed on GPU-capable CI before release.
