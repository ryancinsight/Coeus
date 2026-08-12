# Autograd

Coeus uses dynamic reverse-mode automatic differentiation (autograd) to compute
gradients for tensor operations.

## `Var<T, B>`

`Var<T, B>` is a differentiable variable: a `Tensor<T, B>` plus an optional
gradient accumulator and a link to the backward node that created it.

```rust,ignore
use coeus::autograd::{Var, Parameter};

let x: Var<f32> = Var::from_tensor(data, requires_grad=false);
let w: Parameter<f32> = Parameter::new("weight", weights);
```

## Forward Pass

Every autograd op returns a new `Var` that stores:
- The computed tensor value
- A `BackwardNode` that knows how to propagate gradients

```rust,ignore
let h = coeus::autograd::matmul(&x, &w)?;
let y = h.gelu();
let loss = coeus::autograd::cross_entropy_loss(&y, &targets)?;
```

## Backward Pass

```rust,ignore
loss.backward();  // topological reverse traversal

let dw = w.grad().unwrap();  // gradient of loss w.r.t. w
```

## `NoGradGuard`

Disable gradient recording for inference:

```rust,ignore
let _guard = NoGradGuard::new();
let prediction = model.forward(input)?;  // no backward graph built
```

Or use the functional API: `no_grad_guard()` / `push_no_grad()` / `pop_no_grad()`.

## Multi-Label Margin Loss

`multi_label_margin_loss` (delivered via `feat/mlm-provider`) owns the
pairwise `[N, C, C]` active tensor computation via broadcast, target gather
with safe flattened indexing, masked positive hinge, and target/sibling scatter
backward. CTC remains the sole sequential-DP exception.
