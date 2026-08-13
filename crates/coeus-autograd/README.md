# coeus-autograd

Reverse-mode automatic differentiation for [Coeus](../../README.md).

A tape-based autodiff engine over `Var<T, B>`, the differentiable wrapper
around a [`coeus-tensor`](../coeus-tensor/README.md) `Tensor`.

## What is here

- `Var<T, B>` — a value plus its optional gradient and the node that produced
  it. Graph edges are dynamic: `Var::creator` is
  `Option<Arc<dyn BackwardNode<T, B>>>`, so each differentiable operation costs
  one heap allocation and one virtual call at graph-construction time. Forward
  kernels remain monomorphized.
- The `BackwardNode` trait and its per-operation implementations, covering the
  arithmetic, reduction, shape, convolution, attention, and FFT surfaces.
- `GradBuffer` and `Parameter` for accumulating and holding trainable state.
- Grad-mode guards (`no_grad`) for inference paths.
- Topological `backward()` traversal.

FFT nodes depend on `apollo-fft`, which is a hard, non-optional dependency of
this crate.

## Documentation

API docs: <https://docs.rs/coeus-autograd>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
