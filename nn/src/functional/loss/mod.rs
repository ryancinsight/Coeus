//! Loss functions for neural network training.
//!
//! This module provides various loss functions commonly used in machine learning:
//! - MSE Loss (Mean Squared Error)
//! - Cross-Entropy Loss
//! - NLL Loss (Negative Log Likelihood)
//! - L1 Loss (Mean Absolute Error)
//! - Smooth L1 Loss (Huber Loss)
//! - Binary Cross-Entropy Loss
//! - Binary Cross-Entropy with Logits Loss

pub mod bce;
pub mod cross_entropy;
pub mod mse;
pub mod nll;

// Re-export commonly used loss functions
pub use crate::ops::loss::bce_with_logits_loss;
pub use bce::BCEWithLogitsLoss;
pub use cross_entropy::{cross_entropy_loss as cross_entropy, CrossEntropyLoss};
pub use mse::{mse_loss, MSELoss};
pub use nll::{nll_loss, NLLLoss};
