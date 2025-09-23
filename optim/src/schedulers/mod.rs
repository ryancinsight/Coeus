//! Learning rate schedulers
//!
//! Provides learning rate scheduling algorithms compatible with PyTorch's
//! `torch.optim.lr_scheduler` module.

pub mod cosine_warmrestarts;
pub mod cosinelr;
pub mod cycliclr;
pub mod exponentiallr;
pub mod lambdalr;
pub mod multiplicativelr;
pub mod onecyclelr;
pub mod polynomiallr;
pub mod reducelr;
pub mod steplr;

pub use cosine_warmrestarts::CosineAnnealingWarmRestarts;
pub use cosinelr::CosineAnnealingLR;
pub use cycliclr::{CyclicLR, Mode as CyclicMode};
pub use exponentiallr::ExponentialLR;
pub use lambdalr::LambdaLR;
pub use multiplicativelr::MultiplicativeLR;
pub use onecyclelr::OneCycleLR;
pub use polynomiallr::PolynomialLR;
pub use reducelr::{Mode as ReduceMode, ReduceLROnPlateau, ThresholdMode};
pub use steplr::StepLR;
