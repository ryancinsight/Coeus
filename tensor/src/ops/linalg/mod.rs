//! Linear algebra operations module

mod matmul;
mod matrix_ops;

pub use matmul::matmul;
pub use matrix_ops::{addmm, bmm};
