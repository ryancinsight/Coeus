//! First-order optimizers for the Coeus training stack.
//!
//! # Optimizer trait
//! [`Optimizer<T, B>`](traits::Optimizer) requires `step`, `zero_grad`, `set_lr`,
//! and `clip_grad_norm`. Concrete optimizers own their parameter `Var`s
//! (constructed from a `Vec<Var>`, e.g. `SGD::new`) and update them in place
//! on `step()` via fused [`BackendOps`](coeus_ops::BackendOps) kernel calls;
//! gradient buffers are `Arc`-shared with the autograd graph, so `backward()`
//! populates exactly the gradients the optimizer reads.
//!
//! # Implementations
//! - [`SGD`] — stochastic gradient descent with momentum.
//! - [`Adam`] — adaptive moment estimation (β₁, β₂, ε).
//! - [`AdamW`] — Adam with decoupled weight decay.
//! - [`RMSProp`] — root mean square propagation.
//! - [`AdaGrad`] — adaptive gradient accumulator.
//!
//! # Utilities
//! - [`clip_grad_norm`] — global gradient norm clipping.
//! - [`scheduler`] — [`LrScheduler`] implementations: [`StepDecay`], [`CosineAnneal`], [`LinearWarmup`], [`WarmupCosine`].

mod adagrad;
mod adam;
mod adamw;
pub mod clip;
mod rmsprop;
pub mod scheduler;
mod sgd;
pub mod traits;

pub use adagrad::AdaGrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use clip::clip_grad_norm;
pub use rmsprop::RMSProp;
pub use scheduler::{
    CosineAnneal, LinearWarmup, LrScheduler, SchedulerStrategy, StepDecay, WarmupCosine,
};
pub use sgd::SGD;
pub use traits::Optimizer;
