# coeus-sparse

Sparse tensor storage formats for [Coeus](../../README.md).

This crate is deliberately narrow: it defines the sparse containers and their
construction invariants, and nothing else.

## What is here

- `CooTensor` — coordinate-list storage.
- `CsrTensor` — compressed sparse row storage.
- Validating constructors and accessors for both.

## What is not here

No arithmetic. Sparse matrix-vector and matrix-matrix products live in
[`coeus-ops`](../coeus-ops/README.md) and
[`coeus-leto`](../coeus-leto/README.md).

## Documentation

API docs: <https://docs.rs/coeus-sparse>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
