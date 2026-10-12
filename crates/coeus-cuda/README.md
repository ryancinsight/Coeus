# coeus-cuda

CUDA backend for [Coeus](../../README.md) tensor operations.

Exposes the generic `HephaestusBackend<CudaProvider>` from
[`coeus-hephaestus`](../coeus-hephaestus/README.md): one monomorphized backend
over the CUDA provider's operation bundles, instead of a per-vendor shim
struct. Every Coeus operator resolves through the shared bridge
implementations; this crate only names the device and selects the bundles.

```rust,ignore
use coeus_cuda::{CudaProvider, HephaestusBackend};

let backend = HephaestusBackend::<CudaProvider>::new();
```

## Feature gate

The library binds unconditionally over `hephaestus_cuda::CudaDevice`: without
the `cuda` feature that type is hephaestus-cuda's stub, so the crate compiles
but acquisition fails honestly with an unavailable-device error. The `cuda`
feature enables the real NVRTC + driver backend and gates the on-device
tests and benches:

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
