# coeus-nn

Neural-network layers and operators for [Coeus](../../README.md).

Layers are built on the `Module<T, B>` trait and compose differentiable
`Var` values from [`coeus-autograd`](../coeus-autograd/README.md).

## What is here

- **Linear and convolutional** — `Linear`, `Conv1d`, `Conv2d`, `Conv3d`.
- **Normalization** — `LayerNorm`, `RMSNorm`, `BatchNorm`, `GroupNorm`,
  `InstanceNorm`.
- **Attention and sequence** — multi-head attention with mask support,
  transformer encoder and decoder blocks, and RNN layers.
- **Pooling**, embeddings, and `SwiGLU`.
- **Activations**, including the parametric `PReLU`.
- **Losses**, parameter initialization, and `Sequential` composition.

## Documentation

API docs: <https://docs.rs/coeus-nn>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
