// ── Binary ops module ──

mod kernel;
mod arithmetic;

pub use kernel::{elementwise_binary, elementwise_binary_to};
pub use arithmetic::{add, sub, mul, div, add_assign, sub_assign, mul_assign, div_assign};
