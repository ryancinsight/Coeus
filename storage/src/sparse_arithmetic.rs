//! Sparse tensor arithmetic operations
//!
//! Provides efficient implementations of matrix operations for sparse tensors
//! in CSR, CSC, and COO formats.

use crate::{Result, SparseFormat, CsrStorage, CscStorage, CooStorage, StorageError, Storage};
use alloc::{vec, vec::Vec};
use core::cmp::Ordering;

/// Sparse matrix multiplication trait
pub trait SparseMatMul<T: crate::DataType> {
    /// Multiply sparse matrix with another sparse matrix
    ///
    /// # Arguments
    /// * `other` - The right-hand side matrix
    /// * `result_format` - Preferred format for the result (CSR, CSC, or COO)
    ///
    /// # Returns
    /// Result matrix in COO format (can be converted to other formats)
    ///
    /// # Errors
    /// Returns error if matrix dimensions are incompatible
    fn matmul_sparse(&self, other: &Self, result_format: SparseFormat) -> Result<CooStorage<T>>
    where
        Self: Sized;

    /// Multiply sparse matrix with dense vector
    ///
    /// # Arguments
    /// * `vector` - Dense vector to multiply with
    ///
    /// # Returns
    /// Dense result vector
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    fn matvec_mul(&self, vector: &[T]) -> Result<Vec<T>>;

    /// Multiply sparse matrix with dense matrix (A @ B where A is sparse, B is dense)
    ///
    /// # Arguments
    /// * `dense_matrix` - Dense matrix data in row-major order
    /// * `dense_rows` - Number of rows in dense matrix
    /// * `dense_cols` - Number of columns in dense matrix
    ///
    /// # Returns
    /// Dense result matrix in row-major order
    ///
    /// # Errors
    /// Returns error if matrix dimensions are incompatible
    fn matmul_dense(&self, dense_matrix: &[T], dense_rows: usize, dense_cols: usize) -> Result<Vec<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy;
}

/// Sparse matrix addition trait
pub trait SparseAdd<T: crate::DataType> {
    /// Add two sparse matrices
    ///
    /// # Arguments
    /// * `other` - Matrix to add
    ///
    /// # Returns
    /// Result matrix in COO format for simplicity
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    fn add_sparse(&self, other: &Self) -> Result<CooStorage<T>>;
}

/// Sparse matrix subtraction trait
pub trait SparseSub<T: crate::DataType> {
    /// Subtract two sparse matrices
    ///
    /// # Arguments
    /// * `other` - Matrix to subtract
    ///
    /// # Returns
    /// Result matrix in COO format
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    fn sub_sparse(&self, other: &Self) -> Result<CooStorage<T>>;
}

/// Sparse matrix element-wise multiplication trait
pub trait SparseMul<T: crate::DataType> {
    /// Element-wise multiply two sparse matrices
    ///
    /// # Arguments
    /// * `other` - Matrix to multiply element-wise
    ///
    /// # Returns
    /// Result matrix in COO format
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    fn mul_sparse(&self, other: &Self) -> Result<CooStorage<T>>;
}

/// Sparse matrix element-wise division trait
pub trait SparseDiv<T: crate::DataType> {
    /// Element-wise divide two sparse matrices
    ///
    /// # Arguments
    /// * `other` - Matrix to divide by element-wise
    ///
    /// # Returns
    /// Result matrix in COO format
    ///
    /// # Errors
    /// Returns error if dimensions are incompatible
    fn div_sparse(&self, other: &Self) -> Result<CooStorage<T>>;
}

/// Sparse matrix element-wise operations
pub trait SparseElementWise<T: crate::DataType> {
    /// Apply element-wise operation to sparse matrix
    ///
    /// # Arguments
    /// * `op` - Function to apply to each non-zero element
    ///
    /// # Returns
    /// New sparse matrix with operation applied
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        Self: Sized,
        F: Fn(T) -> T;
}

/// Sparse tensor reduction operations
pub trait SparseReduce<T: crate::DataType + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd> {
    /// Sum all elements (including implicit zeros)
    fn sum(&self) -> T;

    /// Mean of all elements (including implicit zeros)
    fn mean(&self) -> T;

    /// Maximum element value (including implicit zeros)
    fn max(&self) -> T;

    /// Minimum element value (including implicit zeros)
    fn min(&self) -> T;

    /// Sum of all non-zero elements only
    fn sum_nz(&self) -> T;

    /// Count non-zero elements
    fn nnz(&self) -> usize;

    /// Calculate sparsity ratio (0.0 = dense, 1.0 = all zeros)
    fn sparsity(&self) -> f64;
}

/// Sparse matrix transpose operations
pub trait SparseTranspose<T: crate::DataType> {
    /// Transpose sparse matrix
    ///
    /// # Returns
    /// Transposed sparse matrix in COO format
    ///
    /// # Errors
    /// Returns error if transpose operation fails
    fn transpose_sparse(&self) -> Result<CooStorage<T>>;
}

/// Sparse matrix reshape operations
pub trait SparseReshape<T: crate::DataType> {
    /// Reshape sparse matrix to new dimensions
    ///
    /// # Arguments
    /// * `new_shape` - New shape dimensions
    ///
    /// # Returns
    /// Reshaped sparse matrix in COO format
    ///
    /// # Errors
    /// Returns error if total elements don't match or reshape is invalid
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CooStorage<T>>;
}

// CSR Matrix Multiplication Implementation
impl<T: crate::DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy> SparseMatMul<T> for CsrStorage<T> {
    fn matmul_sparse(&self, other: &Self, result_format: SparseFormat) -> Result<CooStorage<T>> {
        // Validate dimensions
        if self.shape().dims()[1] != other.shape().dims()[0] {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().dims()[0] * other.shape().dims()[1],
                actual: self.shape().dims()[1] * other.shape().dims()[0],
            });
        }

        let m = self.shape().dims()[0];
        let k = self.shape().dims()[1];
        let n = other.shape().dims()[1];

        // Use COO format for accumulation (easier for construction)
        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // Standard CSR sparse matrix multiplication
        // For each row i in A
        for i in 0..m {
            let row_start = self.indptr()[i];
            let row_end = self.indptr()[i + 1];

            // For each non-zero element A[i,k] in row i
            for a_idx in row_start..row_end {
                let k = self.indices()[a_idx];
                let a_val = self.as_slice()[a_idx];

                // For each non-zero element B[k,j] in row k of B
                let b_row_start = other.indptr()[k];
                let b_row_end = other.indptr()[k + 1];

                for b_idx in b_row_start..b_row_end {
                    let j = other.indices()[b_idx];
                    let b_val = other.as_slice()[b_idx];

                    // Accumulate C[i,j] += A[i,k] * B[k,j]
                    let c_val = a_val * b_val;
                    result_data.push(c_val);
                    result_row_indices.push(i);
                    result_col_indices.push(j);
                }
            }
        }

        // Return result in COO format
        CooStorage::new(result_data, result_row_indices, result_col_indices, &[m, n])
    }

    fn matvec_mul(&self, vector: &[T]) -> Result<Vec<T>> {
        // Validate dimensions
        if self.shape().dims()[1] != vector.len() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().dims()[1],
                actual: vector.len(),
            });
        }

        let m = self.shape().dims()[0];
        let mut result = vec![T::zero(); m];

        // High-performance CSR matvec multiplication with cache-aware blocking
        // Process in blocks to improve cache locality and enable SIMD vectorization
        const BLOCK_SIZE: usize = 64; // Tune based on cache line size

        // For each row block
        for row_block in (0..m).step_by(BLOCK_SIZE) {
            let row_block_end = (row_block + BLOCK_SIZE).min(m);

            // For each row in the block
            for row in row_block..row_block_end {
                let row_start = self.indptr()[row];
                let row_end = self.indptr()[row + 1];

                // Compute dot product for this row
                // Use manual loop unrolling for better SIMD vectorization potential
                let mut sum = T::zero();
                let mut idx = row_start;

                // Main loop - process 4 elements at a time when possible
                while idx + 4 <= row_end {
                    let col0 = self.indices()[idx];
                    let col1 = self.indices()[idx + 1];
                    let col2 = self.indices()[idx + 2];
                    let col3 = self.indices()[idx + 3];

                    let val0 = self.as_slice()[idx];
                    let val1 = self.as_slice()[idx + 1];
                    let val2 = self.as_slice()[idx + 2];
                    let val3 = self.as_slice()[idx + 3];

                    sum = sum + (val0 * vector[col0]);
                    sum = sum + (val1 * vector[col1]);
                    sum = sum + (val2 * vector[col2]);
                    sum = sum + (val3 * vector[col3]);

                    idx += 4;
                }

                // Handle remaining elements
                while idx < row_end {
                    let col_idx = self.indices()[idx];
                    let val = self.as_slice()[idx];
                    sum = sum + (val * vector[col_idx]);
                    idx += 1;
                }

                result[row] = sum;
            }
        }

        Ok(result)
    }

    fn matmul_dense(&self, dense_matrix: &[T], dense_rows: usize, dense_cols: usize) -> Result<Vec<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        // Validate dimensions: self.cols() must equal dense_rows
        if self.shape().dims()[1] != dense_rows {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().dims()[1],
                actual: dense_rows,
            });
        }

        let sparse_rows = self.shape().dims()[0];
        let mut result = vec![T::zero(); sparse_rows * dense_cols];

        // High-performance sparse-dense matrix multiplication
        // For each row in sparse matrix
        for row in 0..sparse_rows {
            let row_start = self.indptr()[row];
            let row_end = self.indptr()[row + 1];

            // For each non-zero element in this row
            for idx in row_start..row_end {
                let col_idx = self.indices()[idx];
                let val = self.as_slice()[idx];

                // Multiply this sparse element with corresponding dense row
                // and accumulate into result matrix
                let dense_row_start = col_idx * dense_cols;

                // For each column in dense matrix
                for dense_col in 0..dense_cols {
                    let dense_val = dense_matrix[dense_row_start + dense_col];
                    let result_idx = row * dense_cols + dense_col;
                    result[result_idx] = result[result_idx] + (val * dense_val);
                }
            }
        }

        Ok(result)
    }
}

// CSC Matrix Multiplication Implementation
impl<T: crate::DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy> SparseMatMul<T> for CscStorage<T> {
    fn matmul_sparse(&self, other: &Self, result_format: SparseFormat) -> Result<CooStorage<T>> {
        // For CSC, it's often better to convert to CSR and use CSR multiplication
        let self_csr = self.to_csr();
        let other_csr = other.to_csr();
        self_csr.matmul_sparse(&other_csr, SparseFormat::Csr)
    }

    fn matvec_mul(&self, vector: &[T]) -> Result<Vec<T>> {
        // Convert to CSR for easier row-based operations
        let csr = self.to_csr();
        csr.matvec_mul(vector)
    }

    fn matmul_dense(&self, dense_matrix: &[T], dense_rows: usize, dense_cols: usize) -> Result<Vec<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        // Convert to CSR for consistent implementation
        let csr = self.to_csr();
        csr.matmul_dense(dense_matrix, dense_rows, dense_cols)
    }
}

// COO Matrix Multiplication Implementation
impl<T: crate::DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy> SparseMatMul<T> for CooStorage<T> {
    fn matmul_sparse(&self, other: &Self, result_format: SparseFormat) -> Result<CooStorage<T>> {
        // Convert to CSR for multiplication, then back to COO
        let self_csr = self.to_csr();
        let other_csr = other.to_csr();
        self_csr.matmul_sparse(&other_csr, SparseFormat::Csr)
    }

    fn matvec_mul(&self, vector: &[T]) -> Result<Vec<T>> {
        // Convert to CSR for multiplication
        let csr = self.to_csr();
        csr.matvec_mul(vector)
    }

    fn matmul_dense(&self, dense_matrix: &[T], dense_rows: usize, dense_cols: usize) -> Result<Vec<T>>
    where
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        // Convert to CSR for consistent implementation
        let csr = self.to_csr();
        csr.matmul_dense(dense_matrix, dense_rows, dense_cols)
    }
}

// Sparse Addition Implementation (all formats)
impl<T: crate::DataType + core::ops::Add<Output = T> + Copy> SparseAdd<T> for CsrStorage<T> {
    fn add_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        // Convert both to COO and add
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.add_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Add<Output = T> + Copy> SparseAdd<T> for CscStorage<T> {
    fn add_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.add_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Add<Output = T> + Copy> SparseAdd<T> for CooStorage<T> {
    fn add_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        // Validate dimensions
        if self.shape().dims() != other.shape().dims() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().size(),
                actual: other.shape().size(),
            });
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // Add all elements from self
        result_data.extend_from_slice(self.as_slice());
        result_row_indices.extend_from_slice(self.row_indices());
        result_col_indices.extend_from_slice(self.col_indices());

        // Add all elements from other
        result_data.extend_from_slice(other.as_slice());
        result_row_indices.extend_from_slice(other.row_indices());
        result_col_indices.extend_from_slice(other.col_indices());

        // Create result COO matrix
        CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())
    }
}

// Sparse Subtraction Implementation (all formats)
impl<T: crate::DataType + core::ops::Sub<Output = T> + core::ops::Neg<Output = T> + Copy> SparseSub<T> for CsrStorage<T> {
    fn sub_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.sub_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Sub<Output = T> + core::ops::Neg<Output = T> + Copy> SparseSub<T> for CscStorage<T> {
    fn sub_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.sub_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Sub<Output = T> + core::ops::Neg<Output = T> + Copy> SparseSub<T> for CooStorage<T> {
    fn sub_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        // Validate dimensions
        if self.shape().dims() != other.shape().dims() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().size(),
                actual: other.shape().size(),
            });
        }

        // Create a map to combine values at the same positions
        use alloc::collections::BTreeMap;
        let mut position_map: BTreeMap<(usize, usize), T> = BTreeMap::new();

        // Add all elements from self
        for ((&row, &col), &val) in self.row_indices().iter().zip(self.col_indices()).zip(self.as_slice()) {
            position_map.insert((row, col), val);
        }

        // Subtract all elements from other
        for ((&row, &col), &val) in other.row_indices().iter().zip(other.col_indices()).zip(other.as_slice()) {
            let entry = position_map.entry((row, col)).or_insert(T::zero());
            *entry = entry.sub(val);
        }

        // Build result COO, keeping only non-zero elements
        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        for ((row, col), val) in position_map {
            if !val.is_zero() {
                result_data.push(val);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
        }

        CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())
    }
}

// Sparse Element-wise Multiplication Implementation (all formats)
impl<T: crate::DataType + core::ops::Mul<Output = T> + Copy> SparseMul<T> for CsrStorage<T> {
    fn mul_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.mul_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Mul<Output = T> + Copy> SparseMul<T> for CscStorage<T> {
    fn mul_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.mul_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Mul<Output = T> + Copy> SparseMul<T> for CooStorage<T> {
    fn mul_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        // Validate dimensions
        if self.shape().dims() != other.shape().dims() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().size(),
                actual: other.shape().size(),
            });
        }

        // Create maps for efficient lookup of other matrix elements
        use alloc::collections::BTreeMap;
        let mut other_map = BTreeMap::new();
        for ((&row, &col), &val) in other.row_indices().iter()
            .zip(other.col_indices().iter())
            .zip(other.as_slice().iter()) {
            other_map.insert((row, col), val);
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // For each element in self, check if same position exists in other
        for ((&row, &col), &val) in self.row_indices().iter()
            .zip(self.col_indices().iter())
            .zip(self.as_slice().iter()) {
            if let Some(&other_val) = other_map.get(&(row, col)) {
                result_data.push(val * other_val);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
        }

        CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())
    }
}

// Sparse Element-wise Division Implementation (all formats)
impl<T: crate::DataType + core::ops::Div<Output = T> + Copy> SparseDiv<T> for CsrStorage<T> {
    fn div_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.div_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Div<Output = T> + Copy> SparseDiv<T> for CscStorage<T> {
    fn div_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        let other_coo = other.to_coo();
        self_coo.div_sparse(&other_coo)
    }
}

impl<T: crate::DataType + core::ops::Div<Output = T> + Copy> SparseDiv<T> for CooStorage<T> {
    fn div_sparse(&self, other: &Self) -> Result<CooStorage<T>> {
        // Validate dimensions
        if self.shape().dims() != other.shape().dims() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().size(),
                actual: other.shape().size(),
            });
        }

        // Create maps for efficient lookup of other matrix elements
        use alloc::collections::BTreeMap;
        let mut other_map = BTreeMap::new();
        for ((&row, &col), &val) in other.row_indices().iter()
            .zip(other.col_indices().iter())
            .zip(other.as_slice().iter()) {
            other_map.insert((row, col), val);
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // For each element in self, check if same position exists in other
        for ((&row, &col), &val) in self.row_indices().iter()
            .zip(self.col_indices().iter())
            .zip(self.as_slice().iter()) {
            if let Some(&other_val) = other_map.get(&(row, col)) {
                result_data.push(val / other_val);
                result_row_indices.push(row);
                result_col_indices.push(col);
            }
        }

        CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())
    }
}

// Element-wise operations for all sparse formats
impl<T: crate::DataType + Copy> SparseElementWise<T> for CsrStorage<T> {
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        F: Fn(T) -> T,
    {
        let new_data: Vec<T> = self.as_slice().iter().map(|&x| op(x)).collect();
        // Create new CSR with modified data but same structure
        CsrStorage::new(new_data, self.indices().to_vec(), self.indptr().to_vec(), self.shape().dims())
    }
}

impl<T: crate::DataType + Copy> SparseElementWise<T> for CscStorage<T> {
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        F: Fn(T) -> T,
    {
        let new_data: Vec<T> = self.as_slice().iter().map(|&x| op(x)).collect();
        // Create new CSC with modified data but same structure
        CscStorage::new(new_data, self.indices().to_vec(), self.indptr().to_vec(), self.shape().dims())
    }
}

impl<T: crate::DataType + Copy> SparseElementWise<T> for CooStorage<T> {
    fn map_nz<F>(&self, op: F) -> Result<Self>
    where
        F: Fn(T) -> T,
    {
        let new_data: Vec<T> = self.as_slice().iter().map(|&x| op(x)).collect();
        // Create new COO with modified data but same structure
        CooStorage::new(new_data, self.row_indices().to_vec(), self.col_indices().to_vec(), self.shape().dims())
    }
}

// Reduction operations for all sparse formats
impl<T: crate::DataType + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd> SparseReduce<T> for CsrStorage<T> {
    fn sum(&self) -> T {
        self.sum_nz()
    }

    fn mean(&self) -> T {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            T::zero()
        } else {
            let total_sum = <Self as SparseReduce<T>>::sum(self);
            total_sum / T::from(total_elements).unwrap_or(T::one())
        }
    }

    fn max(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x > acc { x } else { acc })
    }

    fn min(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x < acc { x } else { acc })
    }

    fn sum_nz(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn nnz(&self) -> usize {
        self.as_slice().len()
    }

    fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }
}

impl<T: crate::DataType + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd> SparseReduce<T> for CscStorage<T> {
    fn sum(&self) -> T {
        self.sum_nz()
    }

    fn mean(&self) -> T {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            T::zero()
        } else {
            let total_sum = <Self as SparseReduce<T>>::sum(self);
            total_sum / T::from(total_elements).unwrap_or(T::one())
        }
    }

    fn max(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x > acc { x } else { acc })
    }

    fn min(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x < acc { x } else { acc })
    }

    fn sum_nz(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn nnz(&self) -> usize {
        self.as_slice().len()
    }

    fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }
}

impl<T: crate::DataType + num_traits::Zero + core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy + PartialOrd> SparseReduce<T> for CooStorage<T> {
    fn sum(&self) -> T {
        self.sum_nz()
    }

    fn mean(&self) -> T {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            T::zero()
        } else {
            let total_sum = <Self as SparseReduce<T>>::sum(self);
            total_sum / T::from(total_elements).unwrap_or(T::one())
        }
    }

    fn max(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x > acc { x } else { acc })
    }

    fn min(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| if x < acc { x } else { acc })
    }

    fn sum_nz(&self) -> T {
        self.as_slice().iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn nnz(&self) -> usize {
        self.as_slice().len()
    }

    fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }
}

// Sparse Transpose Implementation (all formats)
impl<T: crate::DataType + Copy> SparseTranspose<T> for CsrStorage<T> {
    fn transpose_sparse(&self) -> Result<CooStorage<T>> {
        let coo = self.to_coo();
        coo.transpose_sparse()
    }
}

impl<T: crate::DataType + Copy> SparseTranspose<T> for CscStorage<T> {
    fn transpose_sparse(&self) -> Result<CooStorage<T>> {
        let coo = self.to_coo();
        coo.transpose_sparse()
    }
}

impl<T: crate::DataType + Copy> SparseTranspose<T> for CooStorage<T> {
    fn transpose_sparse(&self) -> Result<CooStorage<T>> {
        let mut transposed_row_indices = Vec::with_capacity(self.nnz());
        let mut transposed_col_indices = Vec::with_capacity(self.nnz());

        // Swap row and column indices
        for (&row, &col) in self.row_indices().iter().zip(self.col_indices()) {
            transposed_row_indices.push(col);
            transposed_col_indices.push(row);
        }

        // Transpose shape: (rows, cols) -> (cols, rows)
        let dims = self.shape().dims();
        let transposed_shape = &[dims[1], dims[0]];

        CooStorage::new(
            self.as_slice().to_vec(),
            transposed_row_indices,
            transposed_col_indices,
            transposed_shape,
        )
    }
}

// Sparse Reshape Implementation (all formats)
impl<T: crate::DataType + Copy> SparseReshape<T> for CsrStorage<T> {
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        self_coo.reshape_sparse(new_shape)
    }
}

impl<T: crate::DataType + Copy> SparseReshape<T> for CscStorage<T> {
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CooStorage<T>> {
        let self_coo = self.to_coo();
        self_coo.reshape_sparse(new_shape)
    }
}

impl<T: crate::DataType + Copy> SparseReshape<T> for CooStorage<T> {
    fn reshape_sparse(&self, new_shape: &[usize]) -> Result<CooStorage<T>> {
        // Validate total elements match
        let new_total_elements = new_shape.iter().product::<usize>();
        let current_total_elements = self.shape().size();

        if new_total_elements != current_total_elements {
            return Err(StorageError::ShapeMismatch {
                expected: current_total_elements,
                actual: new_total_elements,
            });
        }

        // For sparse matrices, we need to remap linear indices to new coordinates
        let mut new_row_indices = Vec::with_capacity(self.nnz());
        let mut new_col_indices = Vec::with_capacity(self.nnz());

        // Convert old (row, col) positions to linear indices, then to new coordinates
        for (&row, &col) in self.row_indices().iter().zip(self.col_indices().iter()) {
            let old_shape = self.shape().dims();
            let linear_idx = if old_shape.len() == 2 {
                row * old_shape[1] + col
            } else {
                // For matrices with more than 2 dimensions, flatten to 2D
                // This is a simplified implementation - full ND reshape would be more complex
                return Err(StorageError::ShapeMismatch {
                    expected: 2,
                    actual: old_shape.len(),
                });
            };

            // Convert linear index back to new coordinates
            if new_shape.len() == 2 {
                let new_row = linear_idx / new_shape[1];
                let new_col = linear_idx % new_shape[1];
                new_row_indices.push(new_row);
                new_col_indices.push(new_col);
            } else {
                return Err(StorageError::ShapeMismatch {
                    expected: 2,
                    actual: new_shape.len(),
                });
            }
        }

        // Create reshaped COO matrix
        CooStorage::new(
            self.as_slice().to_vec(),
            new_row_indices,
            new_col_indices,
            new_shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, SparseFormat};
    use coeus_dtype::float::F32;
    use alloc::vec;

    #[test]
    fn test_csr_matvec_mul() {
        // Create simple 2x2 CSR matrix: [[1, 0], [2, 3]]
        // Non-zeros: (0,0)=1, (1,0)=2, (1,1)=3
        let data = vec![F32::new(1.0), F32::new(2.0), F32::new(3.0)];
        let indices = vec![0, 0, 1];  // column indices
        let indptr = vec![0, 1, 3];   // row pointers
        let csr = CsrStorage::new(data, indices, indptr, &[2, 2]).unwrap();

        // Multiply with vector [1, 1]
        let vector = vec![F32::new(1.0), F32::new(1.0)];
        let result = csr.matvec_mul(&vector).unwrap();

        // Expected: [1*1 + 0*1, 2*1 + 3*1] = [1, 5]
        assert_eq!(result, vec![F32::new(1.0), F32::new(5.0)]);
    }

    #[test]
    fn test_csr_matmul_dense() {
        // Create 2x2 CSR matrix: [[1, 0], [2, 3]]
        let data = vec![F32::new(1.0), F32::new(2.0), F32::new(3.0)];
        let indices = vec![0, 0, 1];
        let indptr = vec![0, 1, 3];
        let csr = CsrStorage::new(data, indices, indptr, &[2, 2]).unwrap();

        // Dense matrix 2x3: [[1, 2, 3], [4, 5, 6]]
        let dense_matrix = vec![F32::new(1.0), F32::new(2.0), F32::new(3.0), F32::new(4.0), F32::new(5.0), F32::new(6.0)];
        let result = csr.matmul_dense(&dense_matrix, 2, 3).unwrap();

        // Expected 2x3 result:
        // Row 0: [1*1, 1*2, 1*3] = [1, 2, 3]
        // Row 1: [2*1 + 3*4, 2*2 + 3*5, 2*3 + 3*6] = [14, 19, 24]
        assert_eq!(result, vec![F32::new(1.0), F32::new(2.0), F32::new(3.0), F32::new(14.0), F32::new(19.0), F32::new(24.0)]);
    }

    #[test]
    fn test_csr_sparsity() {
        // Create 3x3 matrix with 2 non-zeros
        let data = vec![F32::new(1.0), F32::new(2.0)];
        let indices = vec![0, 2];
        let indptr = vec![0, 1, 1, 2];
        let csr = CsrStorage::new(data, indices, indptr, &[3, 3]).unwrap();

        assert_eq!(csr.nnz(), 2);
        assert!((csr.sparsity() - (1.0 - 2.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn test_coo_add_sparse() {
        // Matrix A: [[1, 0], [0, 2]]
        let data_a = vec![F32::new(1.0), F32::new(2.0)];
        let row_a = vec![0, 1];
        let col_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_a, col_a, &[2, 2]).unwrap();

        // Matrix B: [[0, 3], [4, 0]]
        let data_b = vec![F32::new(3.0), F32::new(4.0)];
        let row_b = vec![0, 1];
        let col_b = vec![1, 0];
        let coo_b = CooStorage::new(data_b, row_b, col_b, &[2, 2]).unwrap();

        // Result should have all 4 elements
        let result = coo_a.add_sparse(&coo_b).unwrap();
        assert_eq!(result.nnz(), 4);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_coo_sub_sparse() {
        // Matrix A: [[1, 0], [0, 2]]
        let data_a = vec![F32::new(1.0), F32::new(2.0)];
        let row_a = vec![0, 1];
        let col_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_a, col_a, &[2, 2]).unwrap();

        // Matrix B: [[0, 1], [0, 1]]
        let data_b = vec![F32::new(1.0), F32::new(1.0)];
        let row_b = vec![0, 1];
        let col_b = vec![1, 1];
        let coo_b = CooStorage::new(data_b, row_b, col_b, &[2, 2]).unwrap();

        // Result should have 3 elements: A - B
        let result = coo_a.sub_sparse(&coo_b).unwrap();
        assert_eq!(result.nnz(), 3);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_coo_mul_sparse() {
        // Matrix A: [[1, 0], [0, 2]]
        let data_a = vec![F32::new(1.0), F32::new(2.0)];
        let row_a = vec![0, 1];
        let col_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_a, col_a, &[2, 2]).unwrap();

        // Matrix B: [[0, 3], [4, 0]]
        let data_b = vec![F32::new(3.0), F32::new(4.0)];
        let row_b = vec![0, 1];
        let col_b = vec![1, 0];
        let coo_b = CooStorage::new(data_b, row_b, col_b, &[2, 2]).unwrap();

        // Result should have 0 elements (no overlapping positions)
        let result = coo_a.mul_sparse(&coo_b).unwrap();
        assert_eq!(result.nnz(), 0);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_coo_reshape_sparse() {
        // Matrix A: 2x2 matrix [[1, 0], [0, 2]] -> reshape to 4x1
        let data = vec![F32::new(1.0), F32::new(2.0)];
        let row_indices = vec![0, 1];
        let col_indices = vec![0, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        // Reshape to 4x1 (linearize)
        let result = coo.reshape_sparse(&[4, 1]).unwrap();
        assert_eq!(result.nnz(), 2);
        assert_eq!(result.shape().dims(), &[4, 1]);

        // Check that positions are correctly remapped
        // Original (0,0) -> linear index 0 -> new position (0,0)
        // Original (1,1) -> linear index 3 -> new position (3,0)
        let row_indices_result = result.row_indices();
        let col_indices_result = result.col_indices();
        assert_eq!(row_indices_result[0], 0); // (0,0) -> linear 0 -> (0,0)
        assert_eq!(col_indices_result[0], 0);
        assert_eq!(row_indices_result[1], 3); // (1,1) -> linear 3 -> (3,0)
        assert_eq!(col_indices_result[1], 0);
    }

    #[test]
    fn test_coo_reduce_operations() {
        // Matrix: [[1, 0], [0, 2]] (2x2 matrix with 2 non-zeros)
        let data = vec![F32::new(1.0), F32::new(2.0)];
        let row_indices = vec![0, 1];
        let col_indices = vec![0, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        // Test sum (including implicit zeros): 1 + 0 + 0 + 2 = 3
        assert_eq!(coo.sum(), F32::new(3.0));

        // Test mean: 3.0 / 4 = 0.75
        assert_eq!(coo.mean(), F32::new(0.75));

        // Test max: 2.0
        assert_eq!(coo.max(), F32::new(2.0));

        // Test min: 0.0 (implicit zeros)
        assert_eq!(coo.min(), F32::new(0.0));

        // Test sum_nz: 1 + 2 = 3
        assert_eq!(coo.sum_nz(), F32::new(3.0));

        // Test nnz: 2
        assert_eq!(coo.nnz(), 2);

        // Test sparsity: 1 - (2/4) = 0.5
        assert_eq!(coo.sparsity(), 0.5);
    }

    #[test]
    fn test_coo_transpose_sparse() {
        // Matrix: [[1, 0, 2], [0, 0, 3]] (2x3)
        let data = vec![F32::new(1.0), F32::new(2.0), F32::new(3.0)];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 2];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        // Transpose to (3x2)
        let transposed = coo.transpose_sparse().unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.nnz(), 3);

        // Check transposed values and indices
        let transposed_data = transposed.as_slice();
        let transposed_rows = transposed.row_indices();
        let transposed_cols = transposed.col_indices();

        // Original (0,0)=1 -> Transposed (0,0)=1
        // Original (0,2)=2 -> Transposed (2,0)=2
        // Original (1,2)=3 -> Transposed (2,1)=3
        assert_eq!(transposed_data[0], F32::new(1.0));
        assert_eq!(transposed_rows[0], 0);
        assert_eq!(transposed_cols[0], 0);

        assert_eq!(transposed_data[1], F32::new(2.0));
        assert_eq!(transposed_rows[1], 2);
        assert_eq!(transposed_cols[1], 0);

        assert_eq!(transposed_data[2], F32::new(3.0));
        assert_eq!(transposed_rows[2], 2);
        assert_eq!(transposed_cols[2], 1);
    }
}
