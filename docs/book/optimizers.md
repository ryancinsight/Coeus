# Optimizers

Coeus provides parameter update rules through `coeus-optim`, dispatching
through Hephaestus `StatefulUpdateOps` for GPU-accelerated in-place updates.

## `Optimizer` Trait

```rust,ignore
pub trait Optimizer<T, B> {
    fn step(&mut self) -> Result<(), B::Error>;
    fn zero_grad(&mut self) -> Result<(), B::Error>;
    fn set_lr(&mut self, learning_rate: T);
}
```

## Built-In Optimizers

| Optimizer | Constructor parameters after `params` | Description |
|-----------|---------------------------------------|-------------|
| SGD | `lr, momentum` | Stochastic gradient descent |
| Adam | `lr, beta1, beta2, eps` | Adaptive moment estimation |
| AdamW | `lr, beta1, beta2, eps, weight_decay` | Adam with decoupled weight decay |
| AdaGrad | `lr, eps` | Adaptive per-parameter learning rate |
| RmsProp | `lr, alpha, eps` | RMSProp |

## Usage

```rust,ignore
let mut opt = coeus::optim::Adam::new(
    model.parameters(),
    1e-3,
    0.9,
    0.999,
    1e-8,
)?;

// Training loop
for batch in dataloader {
    opt.zero_grad()?;
    let loss = model_forward_and_loss(&batch)?;
    loss.backward()?;
    opt.step()?;
}
```

## GPU Dispatch

Optimizer updates call `device.stateful_update(&plan, &grad, &mut param, &mut state)`
through Hephaestus, keeping all parameter tensors on-device with zero host transfer.
