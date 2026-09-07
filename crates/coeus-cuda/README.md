# coeus-cuda

CUDA backend for [Coeus](../../README.md) tensor operations.

Implements the [`coeus-hephaestus`](../coeus-hephaestus/README.md) provider
traits against NVIDIA CUDA.

## Feature gate

Compute is behind the `cuda` feature. A default build exposes only the storage
and capability types and implements no mathematical traits:

```sh
cargo nextest run --locked -p coeus-cuda --features cuda
```

The `cuda` feature requires `CUDA_TOOLKIT_PATH` and a working CUDA driver.
Provider failures surface to the caller; a present CUDA provider never silently
downgrades execution to the CPU.

## What is here

- Coeus storage and operation adapters bound to Hephaestus CUDA operation
  markers.
- Provider-owned device acquisition, source generation, compilation caching,
  memory transfers, fills, and kernel launch. Coeus contains no CUDA source,
  driver facade, PTX artifact, or launch/cache implementation.
- Attention, convolution, optimizer, pooling, reduction, and fused operations
  routed through Hephaestus CUDA.

## Documentation

API docs: <https://docs.rs/coeus-cuda>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
