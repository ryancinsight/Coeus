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
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
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
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
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

/// Element-wise addition for CSR matrices
pub fn csr_add_csr_kernel<T>(
    lhs_data: &[T],
    lhs_indices: &[usize],
    lhs_indptr: &[usize],
    rhs_data: &[T],
    rhs_indices: &[usize],
    rhs_indptr: &[usize],
    num_rows: usize,
) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: core::ops::Add<Output = T> + Copy + Default,
{
    let mut out_data = Vec::with_capacity(lhs_data.len() + rhs_data.len());
    let mut out_indices = Vec::with_capacity(lhs_indices.len() + rhs_indices.len());
    let mut out_indptr = Vec::with_capacity(num_rows + 1);
    out_indptr.push(0);

    for row in 0..num_rows {
        let mut i = lhs_indptr[row];
        let i_end = lhs_indptr[row + 1];
        let mut j = rhs_indptr[row];
        let j_end = rhs_indptr[row + 1];

        while i < i_end && j < j_end {
            let col_i = lhs_indices[i];
            let col_j = rhs_indices[j];
            if col_i == col_j {
                out_data.push(lhs_data[i] + rhs_data[j]);
                out_indices.push(col_i);
                i += 1;
                j += 1;
            } else if col_i < col_j {
                out_data.push(lhs_data[i]);
                out_indices.push(col_i);
                i += 1;
            } else {
                out_data.push(rhs_data[j]);
                out_indices.push(col_j);
                j += 1;
            }
        }
        while i < i_end {
            out_data.push(lhs_data[i]);
            out_indices.push(lhs_indices[i]);
            i += 1;
        }
        while j < j_end {
            out_data.push(rhs_data[j]);
            out_indices.push(rhs_indices[j]);
            j += 1;
        }
        out_indptr.push(out_data.len());
    }

    Ok((out_data, out_indices, out_indptr))
}

/// Element-wise subtraction for CSR matrices
pub fn csr_sub_csr_kernel<T>(
    lhs_data: &[T],
    lhs_indices: &[usize],
    lhs_indptr: &[usize],
    rhs_data: &[T],
    rhs_indices: &[usize],
    rhs_indptr: &[usize],
    num_rows: usize,
) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: core::ops::Sub<Output = T> + Copy + Default,
{
    let mut out_data = Vec::with_capacity(lhs_data.len() + rhs_data.len());
    let mut out_indices = Vec::with_capacity(lhs_indices.len() + rhs_indices.len());
    let mut out_indptr = Vec::with_capacity(num_rows + 1);
    out_indptr.push(0);

    for row in 0..num_rows {
        let mut i = lhs_indptr[row];
        let i_end = lhs_indptr[row + 1];
        let mut j = rhs_indptr[row];
        let j_end = rhs_indptr[row + 1];

        while i < i_end && j < j_end {
            let col_i = lhs_indices[i];
            let col_j = rhs_indices[j];
            if col_i == col_j {
                out_data.push(lhs_data[i] - rhs_data[j]);
                out_indices.push(col_i);
                i += 1;
                j += 1;
            } else if col_i < col_j {
                out_data.push(lhs_data[i]);
                out_indices.push(col_i);
                i += 1;
            } else {
                out_data.push(T::default() - rhs_data[j]);
                out_indices.push(col_j);
                j += 1;
            }
        }
        while i < i_end {
            out_data.push(lhs_data[i]);
            out_indices.push(lhs_indices[i]);
            i += 1;
        }
        while j < j_end {
            out_data.push(T::default() - rhs_data[j]);
            out_indices.push(rhs_indices[j]);
            j += 1;
        }
        out_indptr.push(out_data.len());
    }

    Ok((out_data, out_indices, out_indptr))
}

/// Element-wise multiplication for CSR matrices
pub fn csr_mul_csr_kernel<T>(
    lhs_data: &[T],
    lhs_indices: &[usize],
    lhs_indptr: &[usize],
    rhs_data: &[T],
    rhs_indices: &[usize],
    rhs_indptr: &[usize],
    num_rows: usize,
) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: core::ops::Mul<Output = T> + Copy + Default,
{
    let mut out_data = Vec::with_capacity(lhs_indices.len().min(rhs_indices.len()));
    let mut out_indices = Vec::with_capacity(lhs_indices.len().min(rhs_indices.len()));
    let mut out_indptr = Vec::with_capacity(num_rows + 1);
    out_indptr.push(0);

    for row in 0..num_rows {
        let mut i = lhs_indptr[row];
        let i_end = lhs_indptr[row + 1];
        let mut j = rhs_indptr[row];
        let j_end = rhs_indptr[row + 1];

        while i < i_end && j < j_end {
            let col_i = lhs_indices[i];
            let col_j = rhs_indices[j];
            if col_i == col_j {
                out_data.push(lhs_data[i] * rhs_data[j]);
                out_indices.push(col_i);
                i += 1;
                j += 1;
            } else if col_i < col_j {
                i += 1;
            } else {
                j += 1;
            }
        }
        out_indptr.push(out_data.len());
    }

    Ok((out_data, out_indices, out_indptr))
}

/// Element-wise addition for Dense and CSR matrices (result is Dense)
pub fn add_dense_csr_kernel<T>(
    dense: &[T],
    csr_data: &[T],
    csr_indices: &[usize],
    csr_indptr: &[usize],
    num_rows: usize,
    num_cols: usize,
) -> Result<Vec<T>>
where
    T: core::ops::Add<Output = T> + Copy + Default,
{
    let mut out = dense.to_vec();
    for row in 0..num_rows {
        let start = csr_indptr[row];
        let end = csr_indptr[row + 1];
        for idx in start..end {
            let col = csr_indices[idx];
            let val = csr_data[idx];
            out[row * num_cols + col] = out[row * num_cols + col] + val;
        }
    }
    Ok(out)
}

/// Element-wise multiplication for Dense and CSR matrices (result is CSR derivative)
pub fn mul_dense_csr_kernel<T>(
    dense: &[T],
    csr_data: &[T],
    csr_indices: &[usize],
    csr_indptr: &[usize],
    num_rows: usize,
    num_cols: usize,
) -> Result<Vec<T>>
where
    T: core::ops::Mul<Output = T> + Copy + Default,
{
    let mut out_data = Vec::with_capacity(csr_data.len());
    for row in 0..num_rows {
        let start = csr_indptr[row];
        let end = csr_indptr[row + 1];
        for idx in start..end {
            let col = csr_indices[idx];
            let val = csr_data[idx];
            out_data.push(val * dense[row * num_cols + col]);
        }
    }
    Ok(out_data)
}

/// Dense-Sparse matrix multiplication (Dense @ CSR -> Dense)
pub fn matmul_dense_csr_kernel<T>(
    dense: &[T],
    csr_data: &[T],
    csr_indices: &[usize],
    csr_indptr: &[usize],
    m: usize,
    k: usize,
    n: usize,
    result: &mut [T],
) -> Result<()>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
{
    // result[i, j] = sum_l dense[i, l] * csr[l, j]
    // We iterate through Dense rows and CSR sparse elements
    for i in 0..m {
        for l in 0..k {
            let dense_val = dense[i * k + l];
            if dense_val == T::default() { continue; }
            
            let csr_start = csr_indptr[l];
            let csr_end = csr_indptr[l + 1];
            for idx in csr_start..csr_end {
                let j = csr_indices[idx];
                let csr_val = csr_data[idx];
                result[i * n + j] = result[i * n + j] + dense_val * csr_val;
            }
        }
    }
    Ok(())
}

/// Transpose a CSR matrix
pub fn csr_transpose_kernel<T>(
    data: &[T],
    indices: &[usize],
    indptr: &[usize],
    num_rows: usize,
    num_cols: usize,
) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: Copy + Default,
{
    let nnz = data.len();
    let mut count = vec![0usize; num_cols];
    for &idx in indices {
        count[idx] += 1;
    }

    let mut out_indptr = vec![0usize; num_cols + 1];
    let mut current = 0;
    for i in 0..num_cols {
        out_indptr[i] = current;
        current += count[i];
    }
    out_indptr[num_cols] = current;

    let mut out_data = vec![T::default(); nnz];
    let mut out_indices = vec![0usize; nnz];
    
    // Use count as temp storage for current write position
    let mut write_pos = out_indptr[0..num_cols].to_vec();

    for row in 0..num_rows {
        let start = indptr[row];
        let end = indptr[row + 1];
        for offset in start..end {
            let col = indices[offset];
            let val = data[offset];
            let dest = write_pos[col];
            
            out_data[dest] = val;
            out_indices[dest] = row;
            write_pos[col] += 1;
        }
    }

    Ok((out_data, out_indices, out_indptr))
}

/// CSR Matrix-Matrix Multiplication (CSR * CSR)
/// Uses transpose of B for efficiency: A * B = C
pub fn csr_matmul_csr_kernel<T>(
    lhs_data: &[T],
    lhs_indices: &[usize],
    lhs_indptr: &[usize],
    rhs_data: &[T],
    rhs_indices: &[usize],
    rhs_indptr: &[usize],
    m: usize,
    k: usize, // common dim
    n: usize,
) -> Result<(Vec<T>, Vec<usize>, Vec<usize>)>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Default + Copy + PartialEq,
{
    // 1. Transpose RHS to get fast column access (which becomes row access)
    // RHS is (k, n). Transpose is (n, k).
    // B_t rows correspond to B cols.
    let (rhs_t_data, rhs_t_indices, rhs_t_indptr) = csr_transpose_kernel(
        rhs_data, rhs_indices, rhs_indptr, k, n
    )?;

    let mut result_data = Vec::new();
    let mut result_indices = Vec::new();
    let mut result_indptr = Vec::with_capacity(m + 1);
    result_indptr.push(0);

    // A is (m, k). B is (k, n). B_t is (n, k).
    // C(i, j) = dot(row i of A, vector j of B)
    // vector j of B is row j of B_t.

    for row_a in 0..m {
        let a_start = lhs_indptr[row_a];
        let a_end = lhs_indptr[row_a + 1];
        
        // If row A is empty, row C is empty
        if a_start == a_end {
            result_indptr.push(result_data.len());
            continue;
        }

        for row_b_t in 0..n { // Iterate over columns of B (rows of B_t)
            let b_start = rhs_t_indptr[row_b_t];
            let b_end = rhs_t_indptr[row_b_t + 1];

            if b_start == b_end { continue; }

            // Dot product of two sparse vectors
            let mut dot = T::default();
            let mut idx_a = a_start;
            let mut idx_b = b_start;
            let mut has_val = false;

            while idx_a < a_end && idx_b < b_end {
                let col_a = lhs_indices[idx_a];
                let col_b = rhs_t_indices[idx_b];

                if col_a == col_b {
                    dot = dot + lhs_data[idx_a] * rhs_t_data[idx_b];
                    has_val = true;
                    idx_a += 1;
                    idx_b += 1;
                } else if col_a < col_b {
                    idx_a += 1;
                } else {
                    idx_b += 1;
                }
            }

            if has_val && dot != T::default() {
                result_data.push(dot);
                result_indices.push(row_b_t);
            }
        }
        result_indptr.push(result_data.len());
    }

    Ok((result_data, result_indices, result_indptr))
}
