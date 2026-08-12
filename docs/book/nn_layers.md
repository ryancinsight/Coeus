# Neural Network Layers

Coeus's `coeus-nn` crate provides a complete neural network layer library
built on top of the autograd engine.

## `Module` Trait

```rust,ignore
pub trait Module<T, B = MoiraiBackend> {
    fn forward(&self, input: Var<T, B>) -> Result<Var<T, B>, ModuleError>;
    fn parameters(&self) -> Vec<Parameter<T, B>>;
}
```

All layers implement `Module`. The `forward` call records ops in the
autograd graph; calling `backward()` on the output propagates gradients
to all `Parameter` leaves.

## Dense Layers

```rust,ignore
let linear = Linear::new(in_features, out_features, bias=true);
let out = linear.forward(x)?;  // [N, in] -> [N, out]
```

## Convolution

`Conv2d::new(in_channels, out_channels, kernel_size, stride, padding)`

## Normalization

| Layer | Description |
|-------|-------------|
| `LayerNorm` | Normalize over last D dimensions |
| `RMSNorm` | Root mean square normalization |
| `BatchNorm1d/2d/3d` | Batch normalization with running stats |
| `GroupNorm` | Group normalization |

## Attention

`MultiHeadAttention`, `ScaledDotProductAttention`, `TransformerEncoder`,
`TransformerDecoder` — all backed by `hephaestus_core::AttentionOps`.

## Recurrent Layers

`Lstm`, `Gru`, `Rnn`, `Bidirectional<R>`

## Positional Encodings

`SinusoidalEncoding`, `RotaryEmbedding` (RoPE)

## Containers

`Sequential<T, B>` chains layers in order. `StaticSeq` is a compile-time-fixed
container for small networks.

## Activation Modules

`ReLU`, `GeLU`, `GeLUTanh`, `SiLU`, `Sigmoid`, `Tanh`, `ELU`,
`LeakyReLU`, `Mish`, `SwiGlu`, and more — each wraps the corresponding
autograd op as a `Module`.
