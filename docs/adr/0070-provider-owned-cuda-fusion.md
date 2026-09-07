# ADR 0070: Provider-owned CUDA fusion

Status: Accepted  \
Date: 2026-09-04  \
Change class: [arch] [major]  \
Board item: `COEUS-HEPHAESTUS-CUDA-FUSION-001`

## Context

`coeus-cuda` carried fused elementwise and reduction kernels in
`kernels/fuse.rs` and `kernels/reduce.rs`. Those modules generated CUDA
source, owned runtime-rank layout metadata, cached compiled kernels, and
launched them through the CUDA driver. Hephaestus now provides the generic
`CudaFusionOps` implementation with those same provider responsibilities.
Keeping both implementations creates two CUDA backend owners and permits
their source, metadata, cache, and launch contracts to diverge.

## Decision

Coeus adapts its existing `ExprNode` and `Layout` values at one boundary and
delegates fused elementwise and reduction execution to
`hephaestus_cuda::CudaFusionOps`. `coeus-cuda/src/fusion.rs` owns only
consumer-to-provider adaptation:

- expression input names are translated from Coeus `val_N` locals to the
  provider's `input_N` locals;
- Coeus layouts become borrowed provider `DynamicStridedView`s without a
  host round trip or storage copy;
- Coeus retains tensor-shape and axis contracts while provider validation,
  source generation, metadata packing, caching, and launch remain upstream.

The duplicate consumer kernel tree and its unused CUDA source, validation,
cache, and launch paths are deleted. The provider's `CudaFusionScalar` trait
is the single scalar-source and identity contract, including Eunomia reduced
precision types; Coeus does not maintain a second CUDA scalar mapping.

## Alternatives rejected

- Retain the Coeus fused kernels beside the provider path: rejected because it
  preserves duplicate CUDA backend ownership and two cache/launch contracts.
- Copy Hephaestus fusion internals into Coeus: rejected because it recreates
  the same implementation fork under another module path.
- Transfer fused inputs to the CPU: rejected because it violates the
  device-resident CUDA contract and hides a missing provider capability.
- Keep `half` as a direct precision dependency: rejected because Eunomia owns
  the Atlas reduced-precision representation contract.

## Invariants

- Fusion dispatch is monomorphized over the scalar, expression, and provider
  types; no trait object or per-element capability branch is introduced.
- Input and output storage remain on the selected CUDA device.
- Provider errors preserve their typed source context at the Coeus boundary.
- Coeus contains no fused CUDA source generation, layout ABI, pipeline cache,
  or launch implementation after this change.
- The adapter uses borrowed views and provider-owned launch metadata; it does
  not copy tensor storage to host memory.

## Verification

Hephaestus provider contract tests cover runtime-rank elementwise broadcast,
runtime-rank reduction, signed-stride reversal, empty reduction identities
and errors, output injectivity, reduced-precision source preludes, and CUDA
physical dispatch. Coeus's CUDA-enabled package check, focused CUDA tests, and
full package gates run against the locked provider revision `1d3d5df` from
Hephaestus PR #274. The consumer manifest follows the provider's canonical Git
source URLs; the lockfile pins the commits, preventing duplicate same-commit
package identities.

`cargo semver-checks` compares the current package with the pre-change
revision: 189 checks pass and seven intentional removals or bound changes
classify the migration as a major release. The package version remains
unchanged because release versioning is outside this implementation change.
