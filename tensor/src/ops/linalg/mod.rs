//! Linear algebra operations module

mod addmm;
mod addr;
mod bmm;
mod eig;
mod matmul;
mod matrix_exp;
mod mv;
mod qr;
mod cholesky;
mod svd;
// mod matrix_ops; // Deprecated, removed

pub use addmm::addmm;
pub use addr::addr;
pub use bmm::bmm;
pub use eig::{eig, eigh};
pub use matmul::matmul;
pub use matrix_exp::{matrix_exp, matrix_power};
pub use mv::mv;
pub use qr::qr;
pub use cholesky::cholesky;
pub use svd::svd;

// Remove deprecated file later
