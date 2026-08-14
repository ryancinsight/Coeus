# coeus-optim

Optimization algorithms for [Coeus](../../README.md).

Optimizers implement a common `Optimizer` trait and update
[`coeus-autograd`](../coeus-autograd/README.md) parameters through fused
`BackendOps` kernels, so the step runs on whichever backend holds the tensors.

## What is here

- **Optimizers** — SGD, Adam, AdamW, RMSProp, AdaGrad.
- **Gradient clipping** — `clip_grad_norm`.
- **Learning-rate schedulers** — `StepDecay`, `CosineAnneal`, `LinearWarmup`,
  `WarmupCosine`.
- **`least_squares`** — a direct linear least-squares solve.

## Documentation

API docs: <https://docs.rs/coeus-optim>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
