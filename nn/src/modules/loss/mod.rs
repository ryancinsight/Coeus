pub mod bce;
pub mod cross_entropy;
pub mod mse;
pub mod nll;
pub mod l1;
pub mod smooth_l1;
pub mod kl_div;

pub use bce::BCEWithLogitsLoss;
pub use cross_entropy::CrossEntropyLoss;
pub use mse::{mse_loss, MSELoss};
pub use nll::NLLLoss;
pub use l1::L1Loss;
pub use smooth_l1::SmoothL1Loss;
pub use kl_div::KLDivLoss;
