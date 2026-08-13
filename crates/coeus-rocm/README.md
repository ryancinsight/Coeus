# coeus-rocm

ROCm provider wiring for [Coeus](../../README.md).

## Scope

This crate contains **no kernels**. It is the binding layer that connects Coeus
operations to `hephaestus-rocm`, where the actual ROCm compute lives. It
declares the zero-sized `RocmProvider`, a device accessor, and the
[`coeus-hephaestus`](../coeus-hephaestus/README.md) provider trait
implementations that point at the corresponding `hephaestus-rocm` operation
types.

If you are looking for the ROCm kernel implementations, they are in
[hephaestus](https://github.com/ryancinsight/hephaestus), not here.

## Platform gating

Attention and convolution are additionally gated on
`all(feature = "rocm", target_os = "linux")`.

## Documentation

API docs: <https://docs.rs/coeus-rocm>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
