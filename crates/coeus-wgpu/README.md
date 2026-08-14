# coeus-wgpu

WebGPU backend for [Coeus](../../README.md) tensor operations.

Implements the [`coeus-hephaestus`](../coeus-hephaestus/README.md) provider
traits against `wgpu`, giving Coeus a portable GPU path across Vulkan and
Metal.

## What is here

- `WgpuStorage` binding tensor storage to `wgpu::Buffer`.
- WGSL compute kernels for unary, binary, matmul, reduction, pooling,
  unfold/fold, and fused elementwise operations. Shader source is generated as
  Rust string templates parameterized on `T::WGSL_TYPE`, so one kernel body
  serves every supported dtype; there are no standalone `.wgsl` files.
- A pipeline cache so each shader is compiled once per device.

One path falls back to the CPU: the strided key-padding mask in attention. It
is documented at its call site.

## Documentation

API docs: <https://docs.rs/coeus-wgpu>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
