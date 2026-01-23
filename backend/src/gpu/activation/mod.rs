//! GPU activation function primitives (placeholder)

pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub use relu::relu_primitive;
pub use sigmoid::sigmoid_primitive;
pub use tanh::tanh_primitive;