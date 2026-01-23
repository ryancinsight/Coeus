//! Activation backward functions

mod sigmoid;
mod relu;
mod tanh;
mod leaky_relu;
mod gelu;

pub use sigmoid::SigmoidFunction;
pub use relu::ReluFunction;
pub use tanh::TanhFunction;
pub use leaky_relu::LeakyReluFunction;
pub use gelu::GeluFunction;
