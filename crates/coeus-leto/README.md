# coeus-leto

Const-rank dispatch shim adapting Coeus dynamic-rank layouts onto
[Leto](https://github.com/ryancinsight/leto) array kernels.

Coeus tensors carry their rank at runtime; Leto kernels are generic over a
const rank `N`. This crate is the single place that bridges the two, resolving
a runtime rank to a monomorphized Leto call through one bounded `match`. It
implements leto ADR 0002.

Keeping the bridge in one crate means the rank dispatch exists once rather than
being repeated at every call site in [`coeus-ops`](../coeus-ops/README.md).

## What is here

Rank-dispatched adapters for the elementwise, reduction, layout and structural,
linear-algebra, attention, convolution, rotary-embedding, sparse, init, and
stateful-update families, plus `contiguous_values` used by
[`coeus-tensor`](../coeus-tensor/README.md) to materialize strided tensors.

## Documentation

API docs: <https://docs.rs/coeus-leto>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
