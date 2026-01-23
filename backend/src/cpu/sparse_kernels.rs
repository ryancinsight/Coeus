//! Sparse matrix kernels for CPU backend
//!
//! Provides optimized implementations for sparse operations on raw data slices.

use crate::Result;

/// Sparse Matrix-Vector Multiplication (CSR format)
/// y = A * x where A is sparse (CSR), x is dense vector
pub fn spmv_csr_kernel<T>(
    data: &[T],
    indices: &[usize],
    indptr: &[usize],
    vector: &[T],
    result: &mut [T],
    num_rows: usize,
) -> Result<()>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy,
{
    // Validate dimensions
    if indptr.len() != num_rows + 1 {
        return Err(crate::BackendError::InvalidInput(
            "CSR indptr length mismatch".to_string(),
        ));
    }
    if result.len() != num_rows {
        return Err(crate::BackendError::InvalidInput(
            "Result vector length mismatch".to_string(),
        ));
    }

    // High-performance CSR matvec multiplication with cache-aware blocking
    #[allow(clippy::items_after_statements)]
    const BLOCK_SIZE: usize = 64; // Tune based on cache line size

    // For each row block
    for row_block in (0..num_rows).step_by(BLOCK_SIZE) {
        let row_block_end = (row_block + BLOCK_SIZE).min(num_rows);

        // For each row in the block
        for (row, result_item) in result
            .iter_mut()
            .enumerate()
            .take(row_block_end)
            .skip(row_block)
        {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            // Compute dot product for this row
            let mut sum = T::default();
            let mut idx = row_start;

            // Main loop - process 4 elements at a time
            while idx + 4 <= row_end {
                let col0 = indices[idx];
                let col1 = indices[idx + 1];
                let col2 = indices[idx + 2];
                let col3 = indices[idx + 3];

                let val0 = data[idx];
                let val1 = data[idx + 1];
                let val2 = data[idx + 2];
                let val3 = data[idx + 3];

                sum = sum + (val0 * vector[col0]);
                sum = sum + (val1 * vector[col1]);
                sum = sum + (val2 * vector[col2]);
                sum = sum + (val3 * vector[col3]);

                idx += 4;
            }

            // Handle remaining elements
            while idx < row_end {
                let col_idx = indices[idx];
                let val = data[idx];
                sum = sum + (val * vector[col_idx]);
                idx += 1;
            }

            *result_item = sum;
        }
    }

    Ok(())
}

/// Sparse Matrix-Matrix Multiplication (CSR x Dense)
/// C = A * B where A is sparse (CSR), B is dense
pub fn spmm_csr_dense_kernel<T>(
    data: &[T],
    indices: &[usize],
    indptr: &[usize],
    dense_matrix: &[T],
    dense_cols: usize,
    result: &mut [T],
    num_rows: usize,
) -> Result<()>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy,
{
    // High-performance sparse-dense matrix multiplication
    for row in 0..num_rows {
        let row_start = indptr[row];
        let row_end = indptr[row + 1];

        for idx in row_start..row_end {
            let col_idx = indices[idx];
            let val = data[idx];

            let dense_row_start = col_idx * dense_cols;

            for dense_col in 0..dense_cols {
                let dense_val = dense_matrix[dense_row_start + dense_col];
                let result_idx = row * dense_cols + dense_col;
                result[result_idx] = result[result_idx] + (val * dense_val);
            }
        }
    }

    Ok(())
}
