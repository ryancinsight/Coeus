# coeus-core

Core numerical traits and execution primitives for [Coeus](../../README.md).

This is the foundation crate: it defines the vocabulary every other Coeus crate
depends on, and depends on no other Coeus crate itself. It contains no
mathematical kernels.

## What is here

- **Scalar traits** — `Scalar`, `Float`, and `FloatOps`, the dtype abstraction
  every generic kernel is written against.
- **Layout** — the strided `Layout` and `ConstLayout` types: shape, strides,
  offset, contiguity queries, and broadcast/transpose derivation.
- **Storage** — the `Storage` / `CpuStorage` ownership contracts and `SendPtr`.
- **Backends** — `ComputeBackend`, the sealed device abstraction carrying
  associated `Error`, `DeviceBuffer<T>`, `KernelDescriptor`, and
  `DispatchFuture<T>` types plus allocation and host-transfer operations.
  `Backend` is a narrower `unsafe` supertrait requiring `Default` and adding a
  single method, `parallel_for`; its safety contract is that `parallel_for`
  must not return until every closure invocation has completed. Two CPU
  implementations ship here: `SequentialBackend` and `MoiraiBackend`.

## Documentation

API docs: <https://docs.rs/coeus-core>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
