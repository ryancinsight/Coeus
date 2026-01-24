//! Linear algebra operations module

mod addmm;
mod addr;
mod bmm;
mod matmul;
mod mv;
// mod matrix_ops; // Deprecated, removed

pub use addmm::addmm;
pub use addr::addr;
pub use bmm::bmm;
pub use matmul::matmul;
pub use mv::mv;

// Remove deprecated file later
