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

The reduction tests check a row-major `[2, 3]` matrix along axis 1, with
`[2, 1]` keep-dimension reduction output. Sum, product, mean, minimum,
maximum, and the four inclusive prefix/suffix sum/product scans each use
exact expectations derived from their mathematical definitions. Leto and
Metal are checked separately against those values, so agreement between
providers cannot hide a shared error. Product and mean use the float-bound
entry points on both providers.

The Leto check runs without a Metal device. The Metal check requires a
device; set `HEPHAESTUS_METAL_REQUIRE_DEVICE=1` to make device acquisition
failure fail the test, as the macOS provider job does. A device-free local
run verifies the CPU expectations and compilation of the Metal calls,
but does not establish Metal execution correctness.

## Documentation

API docs: <https://docs.rs/coeus-metal>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
