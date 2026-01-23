//! Classification backward functions

mod cross_entropy;
mod nll_loss;
mod softmax;

pub use cross_entropy::CrossEntropyFunction;
pub use nll_loss::NLLLossFunction;
pub use softmax::SoftmaxFunction;
