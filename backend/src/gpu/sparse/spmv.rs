//! GPU SpMV (Sparse Matrix-Vector) kernel
//!
//! Provides GPU-accelerated sparse matrix-vector multiplication.
//! This is one of the most important sparse operations for neural networks.
//! Currently uses CPU fallbacks; will be replaced with wgpu compute shaders.

use storage::{CsrStorage, DenseStorage, Result, StorageError, Storage};
use dtype::DataType;
use std::vec;


/// GPU sparse matrix-vector multiplication kernel
///
/// Computes y = A * x where A is a sparse CSR matrix and x is a dense vector.
///
/// # Performance Notes
/// 
/// GPU SpMV typically uses one of these approaches:
/// - CSR-Vector: One thread per row (good for short rows)
/// - CSR-Adaptive: Dynamically choose based on row density
/// - ELL or COO hybrid for irregular sparsity patterns
pub fn gpu_spmv<T: DataType + Default>(
    matrix: &CsrStorage<T>,
    vector: &DenseStorage<T>,
) -> Result<DenseStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    let (rows, cols) = matrix.dims();
    let vector_data = vector.as_slice();
    
    if vector_data.len() != cols {
        return Err(StorageError::ShapeMismatch {
            expected: cols,
            actual: vector_data.len(),
        });
    }
    
    // CPU fallback: simple SpMV using CSR format
    let mut result = vec![T::zero(); rows];
    for row in 0..rows {
        let start = matrix.indptr()[row];
        let end = matrix.indptr()[row + 1];
        let mut sum = T::zero();
        for idx in start..end {
            let col = matrix.indices()[idx];
            sum = sum + matrix.data()[idx] * vector_data[col];
        }
        result[row] = sum;
    }
    
    DenseStorage::from_vec(result, &[rows])
}

/// GPU sparse matrix-matrix multiplication kernel
///
/// Computes C = A * B where both A and B are sparse CSR matrices.
/// Uses inline implementation to avoid coeus_sparse dependency.
pub fn gpu_spmm<T: DataType + Default>(
    lhs: &CsrStorage<T>,
    rhs: &CsrStorage<T>,
) -> Result<CsrStorage<T>>
where
    T: Copy + crate::num_traits::Zero + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + PartialEq,
{
    // CPU fallback via dense conversion
    // Full wgpu kernel would use sparse-sparse multiplication
    let lhs_dense = lhs.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    
    let (lhs_rows, lhs_cols) = lhs.dims();
    let (rhs_rows, rhs_cols) = rhs.dims();
    
    if lhs_cols != rhs_rows {
        return Err(StorageError::ShapeMismatch {
            expected: lhs_cols,
            actual: rhs_rows,
        });
    }
    
    // Simple matmul via dense
    let lhs_data = lhs_dense.as_slice();
    let rhs_data = rhs_dense.as_slice();
    let mut result = vec![T::zero(); lhs_rows * rhs_cols];
    
    for i in 0..lhs_rows {
        for j in 0..rhs_cols {
            let mut sum = T::zero();
            for k in 0..lhs_cols {
                sum = sum + lhs_data[i * lhs_cols + k] * rhs_data[k * rhs_cols + j];
            }
            result[i * rhs_cols + j] = sum;
        }
    }
    
    let result_dense = DenseStorage::from_vec(result, &[lhs_rows, rhs_cols])?;
    CsrStorage::from_dense(&result_dense)
}
