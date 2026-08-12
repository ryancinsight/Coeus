# Optimizers

Coeus provides parameter update rules through `coeus-optim`, dispatching
through Hephaestus `StatefulUpdateOps` for GPU-accelerated in-place updates.

## `Optimizer` Trait

```rust,ignore
pub trait Optimizer<T, B> {
    fn step(&mut self, parameters: &[Parameter<T, B>]) -> Result<()>;
    fn zero_grad(&self, parameters: &[Parameter<T, B>]);
}
```

## Built-In Optimizers

| Optimizer | Parameters struct | Description |
|-----------|-------------------|-------------|
| SGD | `SgdParameters { lr, momentum, weight_decay, dampening, nesterov }` | Stochastic gradient descent |
| Adam | `AdamParameters { lr, beta1, beta2, epsilon, weight_decay }` | Adaptive moment estimation |
| AdamW | `AdamWParameters { lr, beta1, beta2, epsilon, weight_decay }` | Adam with decoupled weight decay |
| AdaGrad | `AdaGradParameters { lr, epsilon, weight_decay }` | Adaptive per-parameter learning rate |
| RmsProp | `RmsPropParameters { lr, alpha, epsilon, weight_decay, momentum }` | RMSProp |

## Usage

```rust,ignore
let mut opt = coeus::optim::Adam::new(model.parameters(), AdamParameters {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.0,
});

// Training loop
for batch in dataloader {
    opt.zero_grad(&model.parameters());
    let loss = model_forward_and_loss(&batch)?;
    loss.backward();
    opt.step(&model.parameters())?;
}
```

## GPU Dispatch

Optimizer updates call `device.stateful_update(&plan, &grad, &mut param, &mut state)`
through Hephaestus, keeping all parameter tensors on-device with zero host transfer.
