//! Linear Algebra module for Coeus.
//!
//! This crate provides high-level linear algebra operations for Tensors.

pub mod error;
pub mod inverse;
pub mod norm;

// Re-exports
pub mod cholesky;
pub mod det;
pub mod qr;
pub mod solve;
pub mod svd;

// Re-exports
pub use cholesky::Cholesky;
pub use det::Det;
pub use inverse::Inverse;
pub use norm::Norm;
pub use qr::QR;
pub use solve::Solve;
pub use svd::SVD;
