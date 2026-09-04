# coeus-wgpu

WebGPU backend for [Coeus](../../README.md) tensor operations.

Implements the [`coeus-hephaestus`](../coeus-hephaestus/README.md) provider
traits through Hephaestus, giving Coeus a portable GPU path across Vulkan and
Metal without a second device implementation.

## What is here

- `WgpuStorage` binding tensor storage to Hephaestus' provider-owned
  `WgpuBuffer`.
- Coeus-to-provider adapters for tensor shapes, expressions, and layouts.
- Provider dispatch for elementwise, matmul, reduction, pooling, unfold/fold,
  and fused operations. Hephaestus owns WGSL source generation, layout
  metadata, pipeline caching, bind groups, and command submission.

One path falls back to the CPU: the strided key-padding mask in attention. It
is documented at its call site.

## Documentation

API docs: <https://docs.rs/coeus-wgpu>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
