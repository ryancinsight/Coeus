# ADR 0071: Provider-owned accelerator backends

Status: Accepted  \
Date: 2026-09-04  \
Change class: [arch] [major]  \
Board item: `COEUS-HEPHAESTUS-CUDA-FUSION-001`

## Context

Coeus had consumer-side WGPU elementwise kernels and a CUDA driver/NVRTC
facade alongside the generic Hephaestus backend. These paths duplicated
device acquisition, source generation, layout metadata, compilation caching,
memory operations, and launch responsibilities. The duplicate ownership made
backend behavior and precision mapping able to diverge between Coeus and
Hephaestus.

Atlas uses Hephaestus as Coeus's GPU backend. Eunomia owns the reduced-
precision representation contract; Coeus must not introduce a direct `half`
precision dependency.

## Decision

Coeus routes WGPU and CUDA operations through the generic Hephaestus provider
seams. Coeus retains only the consumer concerns required at that boundary:

- tensor shape and expression adaptation;
- conversion of Coeus layouts to borrowed provider views;
- provider-backed storage handles and public Coeus error mapping;
- operation marker selection where the Coeus expression contract requires it.

Hephaestus is the single owner of device acquisition, GPU source generation,
layout metadata ABI, pipeline or compilation caching, transfers, fills, bind
groups, command submission, and kernel launch. Coeus's consumer WGPU kernel
tree, CUDA driver/NVRTC facade, checked-in PTX artifact, and direct CUDA memory
operation calls are removed. No compatibility facade remains for the removed
consumer driver API.

## Alternatives rejected

- Keep local kernels beside provider dispatch: rejected because it preserves
  two backend owners and two cache/source contracts.
- Copy provider internals into Coeus: rejected because copied internals drift
  and violate upstream capability ownership.
- Route missing provider operations through CPU execution: rejected because it
  hides a provider capability gap and breaks device-resident execution.
- Add `half` beside Eunomia: rejected because it forks the Atlas precision
  representation contract.

## Invariants

- GPU data remains in provider-owned storage across operation dispatch.
- Coeus adapters use borrowed layout views and do not copy tensor storage to
  host memory for provider dispatch.
- Provider errors retain their typed source context at the Coeus boundary.
- Backend and scalar variation is represented by generic provider seams, not
  duplicated operation bodies or vendor-named public APIs.
- Coeus contains no GPU source-generation, layout-ABI, cache, driver, or
  launch implementation for WGPU or CUDA.
- Reduced-precision mapping continues through Eunomia and the provider scalar
  contracts; no direct `half` dependency is introduced.

## Verification plan and evidence

The delivery revision must run the locked Coeus WGPU, CUDA, and bridge native
suites, strict Clippy, doctests, formatting, lockfile freshness, and source
audits. The source audit must find no Coeus WGPU kernel tree, CUDA driver
facade, direct CUDA memory calls, direct WGPU dependency, or direct `half`
dependency. Hephaestus provider contract tests remain the upstream behavioral
oracle; optimized accelerator results are compared with the existing Coeus
CPU paths under their documented numerical bounds.

Coeus PR #368 consumes the merged Hephaestus PR #274. Cargo.lock records the
provider revision used by each verification run.

The 2026-09-07 locked API comparison against main's source records no default-
feature break in core or the bridge. CUDA and WGPU require a major release for
the removals and scalar bounds below. CUDA-enabled baseline rustdoc generation
stalls in the removed Cutile binding generator; that surface is compared from
public source, without claiming an automated compatibility verdict.

## Migration

Tensor callers retain `CudaBackend`, `WgpuBackend`, and the public
`evaluate_fused`/`evaluate_fused_reduce` entry points. CUDA computation still
requires the `cuda` feature.

Consumers of `coeus_cuda::driver`, `coeus_cuda::kernels`, `CudaDriver`, or
`get_cuda_context` must use Hephaestus directly for device-level work. Acquire
`hephaestus_cuda::CudaDevice` through its `try_default` method and use the
provider's operation traits; Coeus no longer exposes a separate driver facade.

Replace `<T as coeus_cuda::CudaScalar>::CUDA_TYPE` with
`<T as hephaestus_core::DialectScalar<hephaestus_core::CudaC>>::TYPE_TOKEN`
when generating CUDA source outside Coeus. `CudaScalar` now requires
`hephaestus_cuda::CudaFusionScalar`, and `WgpuScalar` requires
`hephaestus_wgpu::WgpuFusionScalar`. Generic callers must satisfy these provider
bounds; the existing sealed Coeus scalar set remains the supported set.

Replace matches on `CudaBackendError::Fusion` and removed WGPU kernel/layout
error variants with the backend's `Dispatch { operation, source }` variant
for provider failures, or `Validation` for tensor contract failures. The
standalone WGPU `LayoutError` export is removed. Preserve a wildcard arm when
matching these non-exhaustive backend errors.

## Revision 2026-09-07

Integration with main preserves the `StaggeredPairOps` contract introduced by
PR #377. Its provider parameter block carries dimensions but no operand strides
or offsets, so the adapter rejects non-contiguous or offset layouts and unequal
operand shapes before dispatch. Preparation rejects invalid grid spacing before
device acquisition. Value comparisons, typed rejection cases, and unchanged
destination checks cover these boundaries.

Parameterized unary adapters retain the public operation encoding: single
parameters use `f64` bits and paired parameters use two packed `f32` words.
Decoding occurs at the provider's `f32` parameter boundary. The leaky-ReLU
derivative uses the configured slope at either signed zero on every provider.

The elementwise bridge instantiates ranks one through eight. Provider tests
must compile these instantiations, because metadata-only checks do not evaluate
every const-generic assertion. ROCm's metadata and generated address arithmetic
must represent the same accepted layouts; host packing tests establish the ABI,
while execution on an AMD device remains separate behavioral evidence.

Device tests honor the provider's compiled backend set. Four inherited Windows
tests forced DX12 despite the provider compiling Vulkan and Metal; the captured
acquisition error confirms that mismatch. The overrides are removed, and test
guards retain typed acquisition diagnostics. Coeus also advances its direct
Moirai requirements to 0.6 with Leto's matching dependency transition so the
provider corrections resolve without a new Git revision quarantine.
