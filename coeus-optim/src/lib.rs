//! First-order optimizers for the Coeus training stack.
//!
//! # Optimizer trait
//! [`Optimizer<T, B>`](traits::Optimizer) requires `step(&mut self, params: &[&Var<T,B>])` and `zero_grad`;
//! all concrete types implement it via fused [`BackendOps`](coeus_ops::BackendOps) kernel calls.
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
