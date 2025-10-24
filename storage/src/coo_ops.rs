//! COO (Coordinate) specific operations
//!
//! Provides specialized operations and utilities for COO format matrices.

use crate::{Result, CsrStorage, CooStorage, CscStorage, StorageError};
use alloc::{collections::BTreeMap, vec, vec::Vec};

impl<T: crate::DataType> CooStorage<T> {
    /// Get row indices slice
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get column indices slice
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get data values slice
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Count non-zero elements (nnz)
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Calculate sparsity ratio (0.0 = dense, 1.0 = all zeros)
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let total_elements = self.shape().size();
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total_elements as f64)
        }
    }

    /// Sort COO matrix in row-major order
    ///
    /// This can improve performance for certain operations
    pub fn sort(&mut self)
    where
        T: Copy,
    {
        // Create indices for sorting
        let mut indices: Vec<usize> = (0..self.nnz()).collect();

        // Sort by row, then by column
        indices.sort_by(|&a, &b| {
            let row_a = self.row_indices[a];
            let row_b = self.row_indices[b];
            let col_a = self.col_indices[a];
            let col_b = self.col_indices[b];

            match row_a.cmp(&row_b) {
                core::cmp::Ordering::Equal => col_a.cmp(&col_b),
                ord => ord,
            }
        });

        // Reorder data arrays
        let mut new_data = Vec::with_capacity(self.nnz());
        let mut new_row_indices = Vec::with_capacity(self.nnz());
        let mut new_col_indices = Vec::with_capacity(self.nnz());

        for &idx in &indices {
            new_data.push(self.data[idx]);
            new_row_indices.push(self.row_indices[idx]);
            new_col_indices.push(self.col_indices[idx]);
        }

        self.data = new_data;
        self.row_indices = new_row_indices;
        self.col_indices = new_col_indices;
    }

    /// Convert COO to CSR format
    #[must_use]
    pub fn to_csr(&self) -> CsrStorage<T>
    where
        T: Copy,
    {
        let rows = self.shape().dims()[0];
        let mut indptr = vec![0; rows + 1];
        let mut indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        // Count elements per row
        for &row in self.row_indices() {
            indptr[row + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=rows {
            indptr[i] += indptr[i - 1];
        }

        // Create temporary arrays to track insertion positions
        let mut positions = indptr.clone();
        positions.pop(); // Remove last element

        // Fill CSR arrays
        for i in 0..self.nnz() {
            let row = self.row_indices()[i];
            let pos = positions[row];
            indices.push(self.col_indices()[i]);
            data.push(self.data()[i]);
            positions[row] += 1;
        }

        CsrStorage {
            data,
            indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Convert COO to CSC format
    #[must_use]
    pub fn to_csc(&self) -> CscStorage<T>
    where
        T: Copy,
    {
        let cols = self.shape().dims()[1];
        let mut indptr = vec![0; cols + 1];
        let mut indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        // Count elements per column
        for &col in self.col_indices() {
            indptr[col + 1] += 1;
        }

        // Compute cumulative sum for indptr
        for i in 1..=cols {
            indptr[i] += indptr[i - 1];
        }

        // Create temporary arrays to track insertion positions
        let mut positions = indptr.clone();
        positions.pop(); // Remove last element

        // Fill CSC arrays
        for i in 0..self.nnz() {
            let col = self.col_indices()[i];
            let pos = positions[col];
            indices.push(self.row_indices()[i]);
            data.push(self.data()[i]);
            positions[col] += 1;
        }

        CscStorage {
            data,
            indices,
            indptr,
            shape: self.shape.clone(),
        }
    }

    /// Transpose COO matrix
    #[must_use]
    pub fn transpose(&self) -> CooStorage<T>
    where
        T: Copy,
    {
        CooStorage {
            data: self.data.clone(),
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            shape: crate::Shape::new(&[self.shape().dims()[1], self.shape().dims()[0]]).unwrap(),
        }
    }

    /// Sparse-sparse matrix multiplication (C = A @ B)
    ///
    /// This implements efficient sparse-sparse multiplication using COO format.
    /// Time complexity: O(nnz_A * nnz_B) in worst case, but typically much better
    /// with sorted indices and early termination.
    #[must_use]
    pub fn matmul_sparse(&self, other: &CooStorage<T>) -> Result<CooStorage<T>>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero + PartialEq,
    {
        // Validate matrix dimensions
        let a_cols = self.shape().dims()[1];
        let b_rows = other.shape().dims()[0];
        if a_cols != b_rows {
            return Err(StorageError::ShapeMismatch {
                expected: vec![self.shape().dims()[0], other.shape().dims()[1]],
                actual: vec![self.shape().dims()[0], a_cols, b_rows, other.shape().dims()[1]],
            });
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // For efficiency, we need sorted COO matrices
        let mut sorted_self = self.clone();
        sorted_self.sort();

        let mut sorted_other = other.clone();
        sorted_other.sort();

        // Build row-to-indices mapping for self (A)
        let mut a_row_map: alloc::collections::BTreeMap<usize, Vec<usize>> = alloc::collections::BTreeMap::new();
        for (idx, &row) in sorted_self.row_indices().iter().enumerate() {
            a_row_map.entry(row).or_insert_with(Vec::new).push(idx);
        }

        // Build col-to-indices mapping for other (B)
        let mut b_col_map: alloc::collections::BTreeMap<usize, Vec<usize>> = alloc::collections::BTreeMap::new();
        for (idx, &col) in sorted_other.col_indices().iter().enumerate() {
            b_col_map.entry(col).or_insert_with(Vec::new).push(idx);
        }

        // Perform sparse-sparse multiplication
        for (&row_a, a_indices) in &a_row_map {
            for (&col_b, b_indices) in &b_col_map {
                let mut sum = T::zero();

                // Find common columns in A and rows in B (which become the inner dimension)
                for &a_idx in a_indices {
                    let col_a = sorted_self.col_indices()[a_idx];
                    let val_a = sorted_self.data()[a_idx];

                    // Binary search for matching row in B
                    if let Some(b_matching_indices) = b_col_map.get(&col_a) {
                        for &b_idx in b_matching_indices {
                            let val_b = sorted_other.data()[b_idx];
                            sum = sum + val_a * val_b;
                        }
                    }
                }

                // Only store non-zero results to maintain sparsity
                if sum != T::zero() {
                    result_data.push(sum);
                    result_row_indices.push(row_a);
                    result_col_indices.push(col_b);
                }
            }
        }

        let result_shape = &[self.shape().dims()[0], other.shape().dims()[1]];
        CooStorage::new(result_data, result_row_indices, result_col_indices, result_shape)
    }

    /// Sparse-dense matrix multiplication (C = A @ B where A is sparse, B is dense)
    ///
    /// This is more efficient than converting A to dense first.
    /// Time complexity: O(nnz_A * dense_cols)
    #[must_use]
    pub fn matmul_dense(&self, dense: &[T]) -> Result<Vec<T>>
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
    {
        let dense_rows = self.shape().dims()[1];
        let dense_cols = dense.len() / dense_rows;

        if dense.len() % dense_rows != 0 {
            return Err(StorageError::ShapeMismatch {
                expected: vec![dense_rows, dense_cols],
                actual: vec![dense.len()],
            });
        }

        let output_rows = self.shape().dims()[0];
        let mut result = vec![T::zero(); output_rows * dense_cols];

        // Perform sparse-dense multiplication
        for i in 0..self.nnz() {
            let row = self.row_indices()[i];
            let col = self.col_indices()[i];
            let val = self.data()[i];

            // Multiply this sparse element with corresponding dense column
            for dense_col in 0..dense_cols {
                let dense_idx = col * dense_cols + dense_col;
                let result_idx = row * dense_cols + dense_col;
                result[result_idx] = result[result_idx] + val * dense[dense_idx];
            }
        }

        Ok(result)
    }

    /// Element-wise addition with another sparse matrix
    #[must_use]
    pub fn add_sparse(&self, other: &CooStorage<T>) -> Result<CooStorage<T>>
    where
        T: Copy + std::ops::Add<Output = T>,
    {
        if self.shape() != other.shape() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        // For simplicity, concatenate and sort. In production, use a merge algorithm.
        let mut result_data = self.data.clone();
        let mut result_row_indices = self.row_indices.clone();
        let mut result_col_indices = self.col_indices.clone();

        result_data.extend_from_slice(other.data());
        result_row_indices.extend_from_slice(other.row_indices());
        result_col_indices.extend_from_slice(other.col_indices());

        let mut result = CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())?;
        result.sort();
        Ok(result)
    }

    /// Element-wise multiplication with another sparse matrix
    #[must_use]
    pub fn mul_sparse(&self, other: &CooStorage<T>) -> Result<CooStorage<T>>
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        if self.shape() != other.shape() {
            return Err(StorageError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
            });
        }

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // Build coordinate to index mapping for other matrix
        let mut coord_map = alloc::collections::BTreeMap::new();
        for i in 0..other.nnz() {
            let coord = (other.row_indices()[i], other.col_indices()[i]);
            coord_map.insert(coord, i);
        }

        // Find common coordinates and multiply
        for i in 0..self.nnz() {
            let coord = (self.row_indices()[i], self.col_indices()[i]);
            if let Some(&other_idx) = coord_map.get(&coord) {
                let product = self.data()[i] * other.data()[other_idx];
                result_data.push(product);
                result_row_indices.push(coord.0);
                result_col_indices.push(coord.1);
            }
        }

        CooStorage::new(result_data, result_row_indices, result_col_indices, self.shape().dims())
    }

    /// Sum all elements
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Copy + std::ops::Add<Output = T> + num_traits::Zero,
    {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    /// Mean of all elements
    #[must_use]
    pub fn mean(&self) -> T
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Div<Output = T> + num_traits::Zero + num_traits::One + From<u32>,
    {
        if self.nnz() == 0 {
            T::zero()
        } else {
            self.sum() / T::from(self.nnz() as u32)
        }
    }

    /// Maximum element
    #[must_use]
    pub fn max(&self) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)).copied()
    }

    /// Minimum element
    #[must_use]
    pub fn min(&self) -> Option<T>
    where
        T: Copy + PartialOrd,
    {
        self.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_coo_sort() {
        // Create unsorted COO matrix
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![1, 0, 1, 0];
        let col_indices = vec![1, 0, 0, 1];
        let mut coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        coo.sort();

        // Should be sorted by row, then column: (0,0), (0,1), (1,0), (1,1)
        assert_eq!(coo.row_indices(), &[0, 0, 1, 1]);
        assert_eq!(coo.col_indices(), &[0, 1, 0, 1]);
        assert_eq!(coo.data(), &[2.0, 4.0, 3.0, 1.0]);
    }

    #[test]
    fn test_coo_to_csr() {
        // Create COO matrix: [[1, 0, 2], [0, 3, 0]]
        let data = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 2, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 3]).unwrap();

        let csr = coo.to_csr();

        // Verify CSR structure
        assert_eq!(csr.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(csr.indices(), &[0, 2, 1]);
        assert_eq!(csr.indptr(), &[0, 2, 3]);
    }

    #[test]
    fn test_coo_matmul_sparse() {
        // Create COO matrices for multiplication test
        // A = [[1, 0], [0, 2]]
        let data_a = vec![1.0, 2.0];
        let row_indices_a = vec![0, 1];
        let col_indices_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_indices_a, col_indices_a, &[2, 2]).unwrap();

        // B = [[3, 0], [0, 4]]
        let data_b = vec![3.0, 4.0];
        let row_indices_b = vec![0, 1];
        let col_indices_b = vec![0, 1];
        let coo_b = CooStorage::new(data_b, row_indices_b, col_indices_b, &[2, 2]).unwrap();

        let result = coo_a.matmul_sparse(&coo_b).unwrap();

        // Expected: [[3, 0], [0, 8]]
        assert_eq!(result.nnz(), 2);
        assert_eq!(result.data(), &[3.0, 8.0]);
        // Note: order may vary due to sorting
    }

    #[test]
    fn test_coo_matmul_dense() {
        // Create COO matrix: [[1, 0], [0, 2]]
        let data = vec![1.0, 2.0];
        let row_indices = vec![0, 1];
        let col_indices = vec![0, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        // Dense vector: [3, 4]
        let dense = vec![3.0, 4.0];

        let result = coo.matmul_dense(&dense).unwrap();

        // Expected: [3, 8]
        assert_eq!(result, vec![3.0, 8.0]);
    }

    #[test]
    fn test_coo_add_sparse() {
        // Create COO matrices: [[1, 0], [0, 2]] + [[0, 3], [4, 0]]
        let data_a = vec![1.0, 2.0];
        let row_indices_a = vec![0, 1];
        let col_indices_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_indices_a, col_indices_a, &[2, 2]).unwrap();

        let data_b = vec![3.0, 4.0];
        let row_indices_b = vec![0, 1];
        let col_indices_b = vec![1, 0];
        let coo_b = CooStorage::new(data_b, row_indices_b, col_indices_b, &[2, 2]).unwrap();

        let result = coo_a.add_sparse(&coo_b).unwrap();

        // Expected: [[1, 3], [4, 2]]
        assert_eq!(result.nnz(), 4);
    }

    #[test]
    fn test_coo_mul_sparse() {
        // Create COO matrices with overlapping non-zeros
        let data_a = vec![2.0, 3.0];
        let row_indices_a = vec![0, 1];
        let col_indices_a = vec![0, 1];
        let coo_a = CooStorage::new(data_a, row_indices_a, col_indices_a, &[2, 2]).unwrap();

        let data_b = vec![4.0, 5.0];
        let row_indices_b = vec![0, 1];
        let col_indices_b = vec![0, 1];
        let coo_b = CooStorage::new(data_b, row_indices_b, col_indices_b, &[2, 2]).unwrap();

        let result = coo_a.mul_sparse(&coo_b).unwrap();

        // Expected: [[8, 0], [0, 15]]
        assert_eq!(result.nnz(), 2);
        assert_eq!(result.data(), &[8.0, 15.0]);
    }

    #[test]
    fn test_coo_reductions() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row_indices = vec![0, 0, 1, 1];
        let col_indices = vec![0, 1, 0, 1];
        let coo = CooStorage::new(data, row_indices, col_indices, &[2, 2]).unwrap();

        assert_eq!(coo.sum(), 10.0);
        assert_eq!(coo.mean(), 2.5);
        assert_eq!(coo.max(), Some(4.0));
        assert_eq!(coo.min(), Some(1.0));
    }
}
