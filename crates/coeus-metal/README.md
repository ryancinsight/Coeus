# coeus-metal

Metal provider wiring for [Coeus](../../README.md).

## Scope

This crate contains **no kernels** and no `.metal` shader sources. It is the
binding layer that connects Coeus operations to `hephaestus-metal`, where the
actual Metal compute lives. It declares the zero-sized `MetalProvider` and the
[`coeus-hephaestus`](../coeus-hephaestus/README.md) provider trait
implementations that delegate to the corresponding `hephaestus-metal`
operation types.

If you are looking for the Metal kernel implementations, they are in
[hephaestus](https://github.com/ryancinsight/hephaestus), not here.

## Coverage

Narrower than the other backends. Bound operations are `f32`/`i32`/`u32`
elementwise, reductions, scan, cross-entropy, random init, rotate-half, and
stateful update. Attention and convolution are **not** implemented for this
provider.

## Documentation

API docs: <https://docs.rs/coeus-metal>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
