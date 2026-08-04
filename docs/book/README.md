# coeus — Tensors, Autodiff, and NN for Atlas

`coeus` is the tensor, automatic differentiation, neural network, and
optimizer library of the Atlas stack.  It replaces `pytorch`/`jax`/`burn`
for Atlas consumers, built over `leto` (CPU) and `hephaestus` (GPU).

## Design goals

- **Backend-generic** — `Tensor<T, B>` is parametric over both element type
  and execution backend; switching from CPU to GPU is a type parameter change.
- **Zero-cost execution backends** — `SequentialBackend` and `MoiraiBackend`
  are zero-sized unit structs; `MoiraiBackend` routes through the moirai
  work-stealing scheduler without runtime dispatch.
- **Autograd** — the `coeus-autograd` crate implements reverse-mode AD over
  the same tensor type; gradient computation shares the backend dispatch.
- **Optimizer steps** — SGD, Adam, AdamW, RMSprop, Adagrad are zero-overhead
  updates that operate in-place on tensors.

## What this book covers

1. `Tensor<T, B>`, backends, and `from_slice_on` construction.
2. Constructors: `linspace`, `arange`, `zeros`, `eye`, `from_fn`.
3. Layouts, views, and zero-copy transposition.
4. Elementwise and scalar reductions.
5. Matrix multiplication with `matmul`.
6. Convolution (1-D/2-D/3-D).
7. Autograd basics.
8. NN layers (linear, conv, norm, attention).
9. Optimizer steps.
