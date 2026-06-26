// ── Sparse Tensor Operations Module ──

/// Sparse format conversions (COO/CSR/dense interconversion).
pub mod conversions;
/// Sparse matrix operations (SpMM, SpMV, backward passes).
pub mod ops;

pub use conversions::{coo_to_csr, coo_to_dense, csr_to_dense, dense_to_coo, dense_to_csr};
pub use ops::{spmm, spmm_backward_dense, spmm_backward_values, spmv};
