# ADR 0069: Provider-owned WGPU fusion

Status: Accepted  
Date: 2026-09-04  
Change class: [arch] [patch]  
Board item: `COEUS-HEPHAESTUS-WGPU-FUSION-001`

## Context

`coeus-wgpu` carried fused elementwise and reduction kernels in
`kernels/fuse.rs` and `kernels/reduce.rs`. Those modules generated WGSL,
encoded runtime-rank layout metadata, cached pipelines, created bind groups,
and submitted command buffers. Hephaestus now exposes the generic
`WgpuFusionOps` implementation with the same provider responsibilities.
Keeping both implementations makes Coeus a second WGPU backend owner and
allows the two shader and metadata contracts to diverge.

## Decision

Coeus adapts its existing `ExprNode` at one boundary and delegates fused
elementwise and reduction execution to `hephaestus_wgpu::WgpuFusionOps`.
`fusion.rs` owns only Coeus-to-provider adaptation:

- expression input names are translated from Coeus `val_N` locals to the
  provider's `input_N` locals;
- Coeus layouts become borrowed provider `DynamicStridedView`s without a
  host round trip or storage copy;
- Coeus preserves tensor-shape validation and the public empty-axis identity
  and error contracts before provider dispatch.

Hephaestus remains the single owner of WGPU fusion source generation, layout
metadata ABI, pipeline caching, bind-group construction, and submission.
The consumer kernels and their reduction-only validation module are deleted.

## Alternatives rejected

- Retain the Coeus fused kernels beside the provider path: rejected because it
  preserves duplicate backend ownership and a second pipeline/cache contract.
- Copy provider fusion internals into Coeus: rejected because it recreates the
  same implementation fork under a different module path.
- Transfer fused inputs to the CPU and evaluate there: rejected because it
  violates the device-resident backend contract and masks missing provider
  capability.

## Invariants

- Fusion dispatch is monomorphized over the scalar, expression, and provider
  types; no trait object or per-element capability branch is introduced.
- Input and output storage remain on the selected WGPU device.
- Provider errors remain typed Coeus backend errors with their source context.
- Coeus has no fused WGPU pipeline, metadata, or command-submission
  implementation after this change.
- The adapter uses a zero-sized scalar marker and borrowed views; it allocates
  only the provider's required dispatch-owned metadata.

## Verification

The focused and full locked Coeus WGPU suites verify value parity for
elementwise and Sum/Product/Mean/Min/Max reductions, integer reduction typing,
empty-axis contracts, invalid axes, alias rejection, and provider-owned
storage. The exact provider revision is recorded by the active CUDA/WGPU
consolidation ADR; hardware-specific parity remains a hosted-provider concern.
