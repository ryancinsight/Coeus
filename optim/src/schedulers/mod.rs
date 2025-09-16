//! Learning rate schedulers
//!
//! Provides learning rate scheduling algorithms compatible with PyTorch's
//! `torch.optim.lr_scheduler` module.

pub mod cosinelr;
pub mod exponentiallr;
pub mod reducelr;
pub mod steplr;

pub use cosinelr::CosineAnnealingLR;
pub use exponentiallr::ExponentialLR;
pub use reducelr::{Mode as ReduceMode, ReduceLROnPlateau, ThresholdMode};
pub use steplr::StepLR;
