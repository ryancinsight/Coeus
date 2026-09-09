# coeus-wgpu

WebGPU backend for [Coeus](../../README.md) tensor operations.

Implements the [`coeus-hephaestus`](../coeus-hephaestus/README.md) provider
traits through Hephaestus, giving Coeus a portable GPU path across Vulkan and
Metal without a second device implementation.

## What is here

- `HephaestusStorage<WgpuBackend, T>` retaining the provider-owned device
  allocation and detaching cloned storage before writes.
- Coeus-to-provider adapters for tensor shapes, expressions, and layouts.
- Provider dispatch for elementwise, matmul, reduction, pooling, unfold/fold,
  and fused operations. Hephaestus owns WGSL source generation, layout
  metadata, pipeline caching, bind groups, and command submission.

Attention masks remain borrowed provider buffers with explicit layouts.
Device kernels preserve cloned outputs through the shared storage contract.

## Documentation

API docs: <https://docs.rs/coeus-wgpu>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
