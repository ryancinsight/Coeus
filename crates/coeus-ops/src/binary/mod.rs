// ── Binary ops module ──

mod arithmetic;
mod comparison;
mod kernel;

pub use arithmetic::{add, add_assign, div, div_assign, mul, mul_assign, sub, sub_assign};
pub use comparison::{eq, ge, gt, le, lt, ne};
pub use kernel::{elementwise_binary, elementwise_binary_to};
