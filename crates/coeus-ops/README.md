# coeus-ops

Numerical tensor operators for [Coeus](../../README.md).

This is the kernel library and the `BackendOps` dispatch surface: the point
where a tensor operation is resolved to a concrete backend implementation.
Operations are generic over `<T: Scalar, B: ComputeBackend>` and monomorphize
to direct specializations.

## What is here

- Elementwise unary and binary operators.
- `matmul`, batched matmul, and outer product.
- Reductions, scans, `topk`, and norms.
- Convolution: `conv1d`/`conv2d`/`conv3d` and their transposed forms.
- Pooling, including adaptive pooling.
- Embedding, interpolation, and `unfold`/`fold`.
- Scaled dot-product attention.
- A lazy fused-expression DAG for combining elementwise operations.
- Provider-dispatched optimizer steps used by
  [`coeus-optim`](../coeus-optim/README.md).

CPU execution routes through [`coeus-leto`](../coeus-leto/README.md) into the
Leto array kernels.

## Documentation

API docs: <https://docs.rs/coeus-ops>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
