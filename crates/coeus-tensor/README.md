# coeus-tensor

The `Tensor<T, B>` type for [Coeus](../../README.md): strided storage, layout
views, and checkpointing.

This crate owns the tensor container and its layout semantics. It holds no
mathematical kernels — those live in
[`coeus-ops`](../coeus-ops/README.md).

## What is here

- `Tensor<T, B>` with copy-on-write storage shared between views.
- Zero-copy layout manipulation: slicing, transposition, broadcasting, and
  reshaping produce a new layout over the same buffer.
- `to_contiguous()` / `to_contiguous_on()`, which return the receiver unchanged
  when it is already contiguous at offset 0 and otherwise materialize a
  compacted copy through `coeus-leto`.
- Element iterators `iter()` and `iter_mut()`. These require a contiguous
  tensor and assert on a strided one; materialize with `to_contiguous()` first.
- `StateArchive` / `StateDict` checkpointing backed by rkyv.

## Documentation

API docs: <https://docs.rs/coeus-tensor>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
