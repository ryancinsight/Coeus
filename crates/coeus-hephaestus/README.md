# coeus-hephaestus

Generic integration layer between [Coeus](../../README.md) and
[Hephaestus](https://github.com/ryancinsight/hephaestus) device providers.

This crate is vendor-neutral. It owns device storage, host/device transfer,
layout validation, and operation dispatch exactly once, so a new accelerator
does not reimplement any of it. A vendor crate supplies a `HephaestusProvider`
plus the per-operation provider traits and inherits the rest.

## What is here

- `HephaestusProvider` and the per-operation provider trait family:
  elementwise, reduction, attention, convolution, cross-entropy, random init,
  rotate-half, stateful update, and staggered gradient/divergence.
- Device storage and host/device transfer.
- Layout validation and dispatch shared by every vendor backend.

Staggered dispatch accepts matching contiguous rank-three layouts with zero
offsets. Preparation rejects non-finite or non-positive grid spacings before
acquiring a device; dispatch rejects unsupported layouts before submitting work.

## Vendor crates

[`coeus-wgpu`](../coeus-wgpu/README.md),
[`coeus-cuda`](../coeus-cuda/README.md),
[`coeus-rocm`](../coeus-rocm/README.md), and
[`coeus-metal`](../coeus-metal/README.md).

## Documentation

API docs: <https://docs.rs/coeus-hephaestus>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
