# coeus-cuda

CUDA backend for [Coeus](../../README.md) tensor operations.

Implements the [`coeus-hephaestus`](../coeus-hephaestus/README.md) provider
traits against NVIDIA CUDA.

## Feature gate

Compute is behind the `cuda` feature. A default build exposes only the storage
and capability types and implements no mathematical traits:

```sh
cargo test -p coeus-cuda --features cuda
```

The `cuda` feature requires `CUDA_TOOLKIT_PATH` and a working CUDA driver.
Provider failures surface to the caller; a present CUDA provider never silently
downgrades execution to the CPU.

## What is here

- CUDA C kernels for matmul launch, reductions, fused elementwise, 2D and 3D
  max and average pooling, 1D pooling, and unfold/fold, plus a checked-in
  `ptx.ptx`.
- Attention, convolution, and optimizer operations bound to Hephaestus CUDA
  operation markers.

## Documentation

API docs: <https://docs.rs/coeus-cuda>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
