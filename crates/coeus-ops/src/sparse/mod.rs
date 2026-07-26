//! Sparse tensor operations: COO/CSR format conversions and sparse BLAS routines.
//!
//! # Zero-copy allocation policy
//! Functions that write every output element use `alloc_on` (uninitialized
//! allocation); functions that produce a dense output from a *sparse* source
//! (e.g. `coo_to_dense`, `csr_to_dense`) use `zeros_on` because un-referenced
//! positions must remain zero.
//!
//! | Function                  | Output alloc | Reason                               |
//! |---------------------------|--------------|--------------------------------------|
//! | `dense_to_coo`            | `alloc_on`   | writes every indices/values slot     |
//! | `coo_to_dense`            | `zeros_on`   | sparse positions remain 0            |
//! | `coo_to_csr`              | `alloc_on`   | writes all nnz + row offsets         |
//! | `csr_to_dense`            | `zeros_on`   | sparse positions remain 0            |
//! | `spmv`                    | `alloc_on`   | writes every output row              |
//! | `spmm`                    | `alloc_on`   | writes every (row, col) pair         |
//! | `spmm_backward_values`    | `alloc_on`   | writes every nnz gradient            |
//! | `spmm_backward_dense`     | `alloc_on`   | writes every (k, n) pair             |

// ── Sparse Tensor Operations Module ──

/// Sparse format conversions (COO/CSR/dense interconversion).
pub mod conversions;
/// Sparse matrix operations (SpMM, SpMV, backward passes).
pub mod ops;

pub use conversions::{coo_to_csr, coo_to_dense, csr_to_dense, dense_to_coo, dense_to_csr};
pub use ops::{spmm, spmm_backward_dense, spmm_backward_values, spmv};
