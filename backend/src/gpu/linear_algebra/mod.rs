//! GPU linear algebra operations (placeholder)

pub mod matmul;
pub mod transpose;
pub mod decomposition;

pub use matmul::matmul_primitive;
pub use transpose::transpose_primitive;
pub use decomposition::*;