//! Loss functions for neural network training.
//!
//! This module provides various loss functions commonly used in machine learning:
//! - MSE Loss (Mean Squared Error)
//! - Cross-Entropy Loss
//! - NLL Loss (Negative Log Likelihood)
//! - L1 Loss
//! - Smooth L1 Loss
//! - Binary Cross-Entropy with Logits Loss
//! - Focal Loss
//! - Dice Loss
//! - Tversky Loss
//! - KL Divergence Loss
//! - Triplet Margin Loss
//! - Combo Loss

pub mod mse;
pub mod cross_entropy;
pub mod nll;

// Re-export commonly used loss functions
pub use mse::MSELoss;
pub use mse::mse_loss;
pub use cross_entropy::CrossEntropyLoss;
pub use cross_entropy::cross_entropy_loss;
pub use nll::NLLLoss;
pub use nll::nll_loss;
