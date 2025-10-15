//! Sparse tensor implementations.
//!
//! This module contains tensor operations specific to sparse storage formats:
//! CSR, CSC, and COO. These implementations are separated to reduce the size
//! of the main tensor module.

use alloc::vec::Vec;

use crate::{
    error::TensorError,
    Backend, DataType, DenseStorage, Result, Storage, Tensor,
};

// CSR (Compressed Sparse Row) implementations
impl<B, T> Tensor<B, crate::CsrStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Returns the number of non-zero elements in the sparse tensor.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Returns the sparsity ratio (nnz / `total_elements`).
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.storage.sparsity()
    }

    /// Returns the sparse format type.
    #[must_use]
    pub fn sparse_format(&self) -> crate::SparseFormat {
        crate::SparseFormat::Csr
    }

    /// Converts the sparse tensor to dense format.
    ///
    /// This creates a new dense tensor with the same shape and fills in
    /// the non-zero values at their correct positions.
    ///
    /// # Errors
    /// Returns error if dense tensor creation fails.
    pub fn to_dense(&self) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero + Clone,
    {
        let mut dense_data = alloc::vec![T::zero(); self.shape().size()];

        // Fill in non-zero values
        let rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];
        let data = self.as_slice();
        let indices = self.storage.indices();
        let indptr = self.storage.indptr();

        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            for i in start..end {
                let col = indices[i];
                let idx = row * cols + col;
                dense_data[idx] = data[i];
            }
        }

        let dense_storage = DenseStorage::from_vec(dense_data, self.shape().dims())?;
        Ok(Tensor::from_storage(dense_storage, self.backend.clone()))
    }

    /// Performs sparse matrix-vector multiplication (CSR matrix × dense vector).
    ///
    /// For a sparse matrix A (m×n) and dense vector x (n×1), computes y = A×x (m×1).
    /// Only processes non-zero elements for efficiency.
    ///
    /// # Arguments
    /// * `vector` - Dense vector to multiply with (must have shape [n])
    ///
    /// # Returns
    /// Result vector with shape [m]
    ///
    /// # Errors
    /// Returns error if dimensions don't match or operation fails.
    ///
    /// # Examples
    /// ```ignore
    /// # use coeus_tensor::Tensor;
    /// # use coeus_backend::CpuBackend;
    /// # use coeus_dtype::float::Float32;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let backend = CpuBackend::new();
    /// // 2×3 sparse matrix * 3×1 vector = 2×1 result
    /// # let data = vec![Float32::from(1.0), Float32::from(2.0), Float32::from(3.0)];
    /// # let indices = vec![0, 1, 2];
    /// # let indptr = vec![0, 2, 3];
    /// # let sparse = Tensor::from_csr(data, indices, indptr, &[2, 3], backend.clone())?;
    /// # let vector = Tensor::from_vec(vec![Float32::from(1.0), Float32::from(2.0), Float32::from(3.0)], &[3])?;
    /// let result = sparse.matvec_mul(&vector)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn matvec_mul(
        &self,
        vector: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero
            + num_traits::One
            + Clone
            + core::ops::Mul<Output = T>
            + core::ops::Add<Output = T>,
    {
        // Validate dimensions
        if self.shape().ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![2],
                actual: alloc::vec![self.shape().ndim()],
                operation: "matvec_mul",
            });
        }
        if vector.shape().ndim() != 1 {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![1],
                actual: alloc::vec![vector.shape().ndim()],
                operation: "matvec_mul",
            });
        }
        if self.shape().dims()[1] != vector.shape().dims()[0] {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![self.shape().dims()[1]],
                actual: alloc::vec![vector.shape().dims()[0]],
                operation: "matvec_mul",
            });
        }

        let rows = self.shape().dims()[0];
        let mut result_data = alloc::vec![T::zero(); rows];

        let data = self.as_slice();
        let indices = self.storage.indices();
        let indptr = self.storage.indptr();
        let vector_data = vector.as_slice();

        // Sparse matrix-vector multiplication
        for row in 0..rows {
            let start = indptr[row];
            let end = indptr[row + 1];

            for i in start..end {
                let col = indices[i];
                let val = data[i];
                let vec_val = vector_data[col];
                result_data[row] = result_data[row] + (val * vec_val);
            }
        }

        let result_storage = DenseStorage::from_vec(result_data, &[rows])?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Performs sparse matrix multiplication with dense matrix (CSR matrix × dense matrix).
    ///
    /// For a sparse matrix A (m×k) and dense matrix B (k×n), computes C = A×B (m×n).
    /// Only processes non-zero elements of A for efficiency.
    ///
    /// # Arguments
    /// * `dense_matrix` - Dense matrix to multiply with (must have shape [k, n])
    ///
    /// # Returns
    /// Result matrix with shape [m, n]
    ///
    /// # Errors
    /// Returns error if dimensions don't match or operation fails.
    pub fn matmul_dense(
        &self,
        dense_matrix: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero
            + num_traits::One
            + Clone
            + core::ops::Mul<Output = T>
            + core::ops::Add<Output = T>,
    {
        // Validate dimensions
        if self.shape().ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![2],
                actual: alloc::vec![self.shape().ndim()],
                operation: "sparse_matmul",
            });
        }
        if dense_matrix.shape().ndim() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![2],
                actual: alloc::vec![dense_matrix.shape().ndim()],
                operation: "sparse_matmul",
            });
        }
        if self.shape().dims()[1] != dense_matrix.shape().dims()[0] {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![self.shape().dims()[1]],
                actual: alloc::vec![dense_matrix.shape().dims()[0]],
                operation: "sparse_matmul",
            });
        }

        let m = self.shape().dims()[0]; // rows of sparse matrix
        let n = dense_matrix.shape().dims()[1]; // cols of dense matrix

        let mut result_data = alloc::vec![T::zero(); m * n];

        let sparse_data = self.as_slice();
        let sparse_indices = self.storage.indices();
        let sparse_indptr = self.storage.indptr();
        let dense_data = dense_matrix.as_slice();

        // Sparse-dense matrix multiplication
        for i in 0..m {
            // For each row i in sparse matrix
            let row_start = sparse_indptr[i];
            let row_end = sparse_indptr[i + 1];

            for j in 0..n {
                // For each column j in dense matrix
                let mut sum = T::zero();

                for idx in row_start..row_end {
                    let k_idx = sparse_indices[idx];
                    let sparse_val = sparse_data[idx];
                    let dense_val = dense_data[k_idx * n + j];
                    sum = sum + (sparse_val * dense_val);
                }

                result_data[i * n + j] = sum;
            }
        }

        let result_storage = DenseStorage::from_vec(result_data, &[m, n])?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Performs element-wise addition with a dense tensor.
    ///
    /// The result is always dense since adding a scalar to a sparse matrix
    /// can create new non-zero elements.
    ///
    /// # Arguments
    /// * `other` - Dense tensor to add element-wise
    ///
    /// # Returns
    /// Dense tensor with the same shape containing the element-wise sum
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn add_dense(
        &self,
        other: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone + Default,
        T: num_traits::Zero + Clone + core::ops::Add<Output = T>,
    {
        // Validate shapes match
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "sparse_add_dense",
            });
        }

        // Convert sparse to dense first, then add element-wise
        let sparse_dense = self.to_dense()?;
        let sparse_data = sparse_dense.as_slice();
        let other_data = other.as_slice();

        let result_data: Vec<T> = sparse_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        let result_storage = DenseStorage::from_vec(result_data, self.shape().dims())?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Performs element-wise multiplication with a dense tensor.
    ///
    /// Multiplies each element of the sparse tensor by the corresponding
    /// element of the dense tensor. The result preserves the sparse structure.
    ///
    /// # Arguments
    /// * `other` - Dense tensor to multiply element-wise
    ///
    /// # Returns
    /// Sparse tensor with the same sparsity pattern containing element-wise products
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn mul_dense(&self, other: &Tensor<B, DenseStorage<T>, T>) -> Result<Self>
    where
        B: Clone,
        T: Clone + core::ops::Mul<Output = T>,
    {
        // Validate shapes match
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "sparse_mul_dense",
            });
        }

        // Create new sparse data by multiplying existing non-zeros with dense values
        let data = self.as_slice();
        let indices = self.storage.indices();
        let indptr = self.storage.indptr();
        let dense_data = other.as_slice();

        let mut new_data = Vec::with_capacity(data.len());
        let cols = self.shape().dims()[1];

        for row in 0..self.shape().dims()[0] {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            for i in row_start..row_end {
                let col = indices[i];
                let dense_idx = row * cols + col;
                let dense_val = dense_data[dense_idx];
                new_data.push(data[i] * dense_val);
            }
        }

        let new_storage = crate::CsrStorage::new(
            new_data,
            indices.to_vec(),
            indptr.to_vec(),
            self.shape().dims(),
        )?;

        Ok(Tensor::from_storage(new_storage, self.backend.clone()))
    }

    /// Performs element-wise addition with another sparse tensor.
    ///
    /// Adds corresponding elements of two sparse tensors. The result is generally
    /// denser than the inputs due to overlapping sparsity patterns.
    ///
    /// # Arguments
    /// * `other` - Sparse tensor to add element-wise
    ///
    /// # Returns
    /// Dense tensor containing the element-wise sum
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn add_sparse(
        &self,
        other: &Tensor<B, crate::CsrStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: Clone + core::ops::Add<Output = T> + Default,
    {
        // Validate shapes match
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "sparse_add_sparse",
            });
        }

        let rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // Convert both to dense for simplicity (can be optimized later)
        let self_dense = self.to_dense()?;
        let other_dense = other.to_dense()?;
        let self_data = self_dense.as_slice();
        let other_data = other_dense.as_slice();

        // Element-wise addition
        let mut result_data = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            result_data.push(self_data[i] + other_data[i]);
        }

        let result_storage = DenseStorage::from_vec(result_data, &[rows, cols])?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Performs element-wise multiplication with another sparse tensor.
    ///
    /// Multiplies corresponding elements of two sparse tensors. The result preserves
    /// the union of sparsity patterns from both inputs.
    ///
    /// # Arguments
    /// * `other` - Sparse tensor to multiply element-wise
    ///
    /// # Returns
    /// Sparse COO tensor containing the element-wise products
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn mul_sparse(
        &self,
        other: &Tensor<B, crate::CsrStorage<T>, T>,
    ) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone + core::ops::Mul<Output = T> + Default + PartialEq,
    {
        use alloc::collections::BTreeMap;

        // Validate shapes match
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "sparse_mul_sparse",
            });
        }

        let rows = self.shape().dims()[0];
        let cols = self.shape().dims()[1];

        // Convert both to COO format for element-wise operations
        let self_coo = self.to_sparse(&crate::SparseFormat::Coo)?;
        let other_coo = other.to_sparse(&crate::SparseFormat::Coo)?;

        // Create lookup maps for efficient element access
        let mut self_map = BTreeMap::new();
        for i in 0..self_coo.nnz() {
            let key = (
                self_coo.storage.row_indices()[i],
                self_coo.storage.col_indices()[i],
            );
            self_map.insert(key, self_coo.storage.as_slice()[i]);
        }

        let mut other_map = BTreeMap::new();
        for i in 0..other_coo.nnz() {
            let key = (
                other_coo.storage.row_indices()[i],
                other_coo.storage.col_indices()[i],
            );
            other_map.insert(key, other_coo.storage.as_slice()[i]);
        }

        // Compute element-wise product where both tensors have non-zeros
        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // Iterate through all positions where both tensors have non-zeros
        for (&(row, col), &self_val) in &self_map {
            if let Some(&other_val) = other_map.get(&(row, col)) {
                let product = self_val * other_val;
                // Only include non-zero results
                if product != T::default() {
                    result_data.push(product);
                    result_row_indices.push(row);
                    result_col_indices.push(col);
                }
            }
        }

        Tensor::from_coo(
            result_data,
            result_row_indices,
            result_col_indices,
            &[rows, cols],
            self.backend.clone(),
        )
    }

    /// Transposes the sparse tensor (rows become columns, columns become rows).
    #[must_use]
    pub fn transpose(&self) -> Tensor<B, crate::CsrStorage<T>, T>
    where
        B: Clone,
        T: Clone,
    {
        let transposed_storage = self.storage.transpose();
        Tensor::from_storage(transposed_storage, self.backend.clone())
    }

    /// Computes the sum of all non-zero elements in the sparse tensor.
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T>,
    {
        self.as_slice()
            .iter()
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    /// Computes the mean of all non-zero elements in the sparse tensor.
    ///
    /// Note: This computes the mean of non-zero elements only, not including zeros.
    #[must_use]
    pub fn mean(&self) -> f64
    where
        T: Into<f64>,
    {
        #[allow(clippy::cast_precision_loss)]
        let nnz = self.nnz() as f64;
        if nnz == 0.0 {
            0.0
        } else {
            let sum: f64 = self.as_slice().iter().map(|x| (*x).into()).sum();
            sum / nnz
        }
    }

    /// Converts CSR sparse tensor to COO format.
    ///
    /// # Errors
    /// Returns `TensorError` if conversion fails.
    pub fn to_sparse(&self, format: &crate::SparseFormat) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone,
    {
        match format {
            crate::SparseFormat::Coo => {
                let coo_storage = self.storage.to_coo();
                // Extract data from storage (we need to clone since fields are private)
                let data = coo_storage.as_slice().to_vec();
                let row_indices = coo_storage.row_indices().to_vec();
                let col_indices = coo_storage.col_indices().to_vec();
                Tensor::from_coo(
                    data,
                    row_indices,
                    col_indices,
                    self.shape().dims(),
                    self.backend.clone(),
                )
            }
            crate::SparseFormat::Csr => {
                // Already in CSR format, convert to COO and back
                let coo = self.to_sparse(&crate::SparseFormat::Coo)?;
                coo.to_sparse(&crate::SparseFormat::Csr)
            }
            crate::SparseFormat::Csc => {
                // Convert to CSC via COO
                let coo = self.to_sparse(&crate::SparseFormat::Coo)?;
                coo.to_sparse(&crate::SparseFormat::Csc)
            }
        }
    }
}

// CSC (Compressed Sparse Column) implementations
impl<B, T> Tensor<B, crate::CscStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Returns the number of non-zero elements in the sparse tensor.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Returns the sparsity ratio (nnz / `total_elements`).
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.storage.sparsity()
    }

    /// Returns the sparse format type.
    #[must_use]
    pub fn sparse_format(&self) -> crate::SparseFormat {
        crate::SparseFormat::Csc
    }

    /// Converts the sparse tensor to dense format.
    ///
    /// This creates a new dense tensor with the same shape and fills in
    /// the non-zero values at their correct positions.
    ///
    /// # Errors
    /// Returns error if dense tensor creation fails.
    pub fn to_dense(&self) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero + Clone,
    {
        let mut dense_data = alloc::vec![T::zero(); self.shape().size()];

        // Fill in non-zero values
        let cols = self.shape().dims()[1];
        let data = self.as_slice();
        let indices = self.storage.indices();
        let indptr = self.storage.indptr();

        for col in 0..cols {
            let start = indptr[col];
            let end = indptr[col + 1];

            for i in start..end {
                let row = indices[i];
                let idx = row * cols + col;
                dense_data[idx] = data[i];
            }
        }

        let dense_storage = DenseStorage::from_vec(dense_data, self.shape().dims())?;
        Ok(Tensor::from_storage(dense_storage, self.backend.clone()))
    }

    /// Transposes the sparse tensor (rows become columns, columns become rows).
    #[must_use]
    pub fn transpose(&self) -> Tensor<B, crate::CscStorage<T>, T>
    where
        B: Clone,
        T: Clone,
    {
        let transposed_storage = self.storage.transpose();
        Tensor::from_storage(transposed_storage, self.backend.clone())
    }

    /// Computes the sum of all non-zero elements in the sparse tensor.
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T>,
    {
        self.as_slice()
            .iter()
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    /// Computes the mean of all non-zero elements in the sparse tensor.
    ///
    /// Note: This computes the mean of non-zero elements only, not including zeros.
    #[must_use]
    pub fn mean(&self) -> f64
    where
        T: Into<f64>,
    {
        #[allow(clippy::cast_precision_loss)]
        let nnz = self.nnz() as f64;
        if nnz == 0.0 {
            0.0
        } else {
            let sum: f64 = self.as_slice().iter().map(|x| (*x).into()).sum();
            sum / nnz
        }
    }

    /// Converts CSC sparse tensor to COO format.
    ///
    /// # Errors
    /// Returns `TensorError` if conversion fails.
    pub fn to_sparse(&self, format: &crate::SparseFormat) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone,
    {
        match format {
            crate::SparseFormat::Csc => {
                let coo_storage = self.storage.to_coo();
                let data = coo_storage.as_slice().to_vec();
                let row_indices = coo_storage.row_indices().to_vec();
                let col_indices = coo_storage.col_indices().to_vec();
                Tensor::from_coo(
                    data,
                    row_indices,
                    col_indices,
                    self.shape().dims(),
                    self.backend.clone(),
                )
            }
            crate::SparseFormat::Csr => {
                // Convert via COO
                let coo = self.to_sparse(&crate::SparseFormat::Csc)?;
                coo.to_sparse(&crate::SparseFormat::Csr)
            }
            crate::SparseFormat::Coo => {
                // Convert to COO
                let coo_storage = self.storage.to_coo();
                let data = coo_storage.as_slice().to_vec();
                let row_indices = coo_storage.row_indices().to_vec();
                let col_indices = coo_storage.col_indices().to_vec();
                Tensor::from_coo(
                    data,
                    row_indices,
                    col_indices,
                    self.shape().dims(),
                    self.backend.clone(),
                )
            }
        }
    }
}

// COO (Coordinate) implementations
impl<B, T> Tensor<B, crate::CooStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Returns the number of non-zero elements in the sparse tensor.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Returns the sparsity ratio (nnz / `total_elements`).
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        self.storage.sparsity()
    }

    /// Returns the sparse format type.
    #[must_use]
    pub fn sparse_format(&self) -> crate::SparseFormat {
        crate::SparseFormat::Coo
    }

    /// Sorts the COO tensor by row, then by column for efficient operations.
    pub fn sort(&mut self) {
        self.storage.sort();
    }

    /// Converts the sparse tensor to dense format.
    ///
    /// This creates a new dense tensor with the same shape and fills in
    /// the non-zero values at their correct positions.
    ///
    /// # Errors
    /// Returns error if dense tensor creation fails.
    pub fn to_dense(&self) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero + Clone,
    {
        let mut dense_data = alloc::vec![T::zero(); self.shape().size()];

        // Fill in non-zero values
        let cols = self.shape().dims()[1];
        let data = self.as_slice();
        let row_indices = self.storage.row_indices();
        let col_indices = self.storage.col_indices();

        for i in 0..self.nnz() {
            let row = row_indices[i];
            let col = col_indices[i];
            let idx = row * cols + col;
            dense_data[idx] = data[i];
        }

        let dense_storage = DenseStorage::from_vec(dense_data, self.shape().dims())?;
        Ok(Tensor::from_storage(dense_storage, self.backend.clone()))
    }

    /// Transposes the sparse tensor (rows become columns, columns become rows).
    #[must_use]
    pub fn transpose(&self) -> Tensor<B, crate::CooStorage<T>, T>
    where
        B: Clone,
        T: Clone,
    {
        let transposed_storage = self.storage.transpose();
        Tensor::from_storage(transposed_storage, self.backend.clone())
    }

    /// Computes the sum of all non-zero elements in the sparse tensor.
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T>,
    {
        self.as_slice()
            .iter()
            .copied()
            .fold(T::default(), |acc, x| acc + x)
    }

    /// Computes the mean of all non-zero elements in the sparse tensor.
    ///
    /// Note: This computes the mean of non-zero elements only, not including zeros.
    #[must_use]
    pub fn mean(&self) -> f64
    where
        T: Into<f64>,
    {
        #[allow(clippy::cast_precision_loss)]
        let nnz = self.nnz() as f64;
        if nnz == 0.0 {
            0.0
        } else {
            let sum: f64 = self.as_slice().iter().map(|x| (*x).into()).sum();
            sum / nnz
        }
    }

    /// Converts COO sparse tensor to different sparse formats.
    ///
    /// # Errors
    /// Returns `TensorError` if conversion fails.
    pub fn to_sparse(&self, format: &crate::SparseFormat) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone,
    {
        match format {
            crate::SparseFormat::Coo => {
                // Already COO, clone
                let data = self.storage.as_slice().to_vec();
                let row_indices = self.storage.row_indices().to_vec();
                let col_indices = self.storage.col_indices().to_vec();
                Tensor::from_coo(
                    data,
                    row_indices,
                    col_indices,
                    self.shape().dims(),
                    self.backend.clone(),
                )
            }
            crate::SparseFormat::Csr => {
                let csr_storage = self.storage.to_csr();
                let data = csr_storage.as_slice().to_vec();
                let indices = csr_storage.indices().to_vec();
                let indptr = csr_storage.indptr().to_vec();
                let tensor = Tensor::from_csr(
                    data,
                    indices,
                    indptr,
                    self.shape().dims(),
                    self.backend.clone(),
                )?;
                // Convert back to COO for return type
                tensor.to_sparse(&crate::SparseFormat::Coo)
            }
            crate::SparseFormat::Csc => {
                let csc_storage = self.storage.to_csc();
                let data = csc_storage.as_slice().to_vec();
                let indices = csc_storage.indices().to_vec();
                let indptr = csc_storage.indptr().to_vec();
                let tensor = Tensor::from_csc(
                    data,
                    indices,
                    indptr,
                    self.shape().dims(),
                    self.backend.clone(),
                )?;
                // Convert back to COO for return type
                tensor.to_sparse(&crate::SparseFormat::Coo)
            }
        }
    }
}
