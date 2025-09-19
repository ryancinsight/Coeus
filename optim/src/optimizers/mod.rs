//! Specific optimizer implementations
//!
//! Contains implementations of popular optimization algorithms
//! compatible with PyTorch's torch.optim module.

pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod asgd;
pub mod lbfgs;
pub mod rmsprop;
pub mod rprop;
pub mod sgd;
pub mod sparse_adam;

pub use adagrad::Adagrad;
pub use adam::Adam;
pub use adamw::AdamW;
pub use asgd::Asgd;
pub use lbfgs::LBFGS;
pub use rmsprop::Rmsprop;
pub use rprop::Rprop;
pub use sgd::Sgd;
pub use sparse_adam::SparseAdam;
