//! Classification operations module

mod cross_entropy;
mod log_softmax;
mod nll_loss;
mod softmax;

pub use cross_entropy::cross_entropy;
pub use log_softmax::log_softmax;
pub use nll_loss::nll_loss;
pub use softmax::softmax;
