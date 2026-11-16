//! Linear-algebra autograd operations.

mod dense;
mod sparse;

pub use dense::{matmul, transpose_2d, MatmulOp, Transpose2dOp};
pub use sparse::{sparse_matmul, sparse_matmul_coo, SparseCooMatMulNode, SparseMatMulNode};
