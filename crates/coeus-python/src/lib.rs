// ── Coeus Python bindings entry point ──
#![deny(missing_docs)]
//! PyO3 Python bindings for the Coeus tensor library.
#[cfg(feature = "mnemosyne-global")]
#[global_allocator]
static GLOBAL: mnemosyne::Mnemosyne = mnemosyne::Mnemosyne;

/// Activation functions (ReLU, GELU, SiLU, etc.).
pub mod activations;
/// Distributed training communicators.
pub mod dist;
mod error;
mod grad_mode;
/// Weight initialization functions (uniform, normal, xavier, kaiming).
pub mod init;
/// Loss functions (MSE, cross-entropy, etc.).
pub mod losses;
/// Neural network layers (Linear, Conv, Norm, Attention, Transformer, RNN, Pool, etc.).
pub mod nn;
/// Tensor operations (arithmetic, reductions, shaping, functional nn).
pub mod ops;
/// Optimizers and learning rate schedulers.
pub mod optim;
/// Tensor type and supporting utilities (state dict, iterator).
pub mod tensor;

mod api;
mod grad_ctx;

pub use api::{pycoeus, shutdown};
pub use grad_ctx::NoGradCtx;
