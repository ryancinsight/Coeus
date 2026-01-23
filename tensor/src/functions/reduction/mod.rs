//! Reduction backward functions

mod sum;
mod mean;
mod max;
mod min;

pub use sum::SumFunction;
pub use mean::MeanFunction;
pub use max::MaxFunction;
pub use min::MinFunction;
