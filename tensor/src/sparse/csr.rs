use crate::{error::TensorError, Backend, DataType, DenseStorage, Result, Tensor};
use alloc::vec::Vec;

// CSR (Compressed Sparse Row) implementations
impl<B, T> Tensor<B, crate::CsrStorage<T>, T>
where
    B: Backend<Data = T>,
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

    // Note: to_dense() is defined in ops/sparse/mod.rs to avoid duplication

    /// Performs sparse matrix-vector multiplication (CSR matrix × dense vector).
    ///
    /// For a sparse matrix A (m×n) and dense vector x (n×1), computes y = A×x (m×1).
    /// Only processes non-zero elements for efficiency.
    ///
    /// # Arguments
    /// * `vector` - Dense vector tensor with shape \[n, 1\] or \[n\]
    ///
    /// # Returns
    /// Result vector with shape \[m, 1\] or \[m\] (matching input vector rank)
    ///
    /// # Errors
    /// Returns error if dimensions don't match or operation fails.
    pub fn matvec_mul(
        &self,
        vector: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero + Clone + core::ops::AddAssign + core::ops::Mul<Output = T> + Default + 'static,
    {
        
        
        let dims = self.shape().dims();
        let m = dims[0];
        let n = dims[1];

        // Check dimensions
        let vec_dims = vector.shape().dims();
        if vec_dims[0] != n {
            return Err(TensorError::ShapeMismatch {
                expected: vec![n, 1],
                actual: vec_dims.to_vec(),
                operation: "matvec_mul",
            });
        }

        // Inline SpMV: y = A * x where A is sparse (CSR), x is dense vector
        let vec_data = vector.as_slice();
        let mut result_data = alloc::vec![T::zero(); m];
        
        let indptr = self.storage.indptr();
        let indices = self.storage.indices();
        let data_slice = self.storage.data();

        // For each row of A
        for i in 0..m {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            
            let mut sum = T::zero();
            for idx in row_start..row_end {
                let col = indices[idx];
                let a_val = data_slice[idx].clone();
                let x_val = vec_data[col].clone();
                sum += a_val * x_val;
            }
            result_data[i] = sum;
        }

        // Output shape matches input vector rank
        let result_shape = if vec_dims.len() == 1 {
            alloc::vec![m]
        } else {
            alloc::vec![m, 1]
        };

        let result_storage = DenseStorage::from_vec(result_data, &result_shape).map_err(TensorError::StorageError)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }

    /// Performs sparse matrix multiplication with dense matrix (CSR matrix × dense matrix).
    ///
    /// For a sparse matrix A (m×k) and dense matrix B (k×n), computes C = A×B (m×n).
    /// Only processes non-zero elements of A for efficiency.
    ///
    /// # Arguments
    /// * `dense_matrix` - Dense matrix tensor with shape [k, n]
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
        T: num_traits::Zero + Clone + core::ops::AddAssign + core::ops::Mul<Output = T> + Default + 'static,
    {
        let lhs_dims = self.shape().dims();
        if lhs_dims.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: lhs_dims.len(),
                message: format!("Left matrix must be 2D, got shape {lhs_dims:?}"),
            });
        }
        let m = lhs_dims[0];
        let k = lhs_dims[1];

        let rhs_dims = dense_matrix.shape().dims();
        if rhs_dims.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: rhs_dims.len(),
                message: format!("Right matrix must be 2D, got shape {rhs_dims:?}"),
            });
        }
        if rhs_dims[0] != k {
            return Err(TensorError::ShapeError {
                expected: k,
                actual: rhs_dims[0],
                message: format!(
                    "Matrix dimension mismatch: {}×{} @ {}×{} (inner dimensions {} ≠ {})",
                    m, k, rhs_dims[0], rhs_dims[1], k, rhs_dims[0]
                ),
            });
        }
        let n = rhs_dims[1];

        // Inline SpMM: C = A × B where A is sparse (CSR), B is dense
        // Result C is m×n dense matrix
        let dense_data = dense_matrix.as_slice();
        let mut result = alloc::vec![T::zero(); m * n];
        
        let indptr = self.storage.indptr();
        let indices = self.storage.indices();
        let data = self.storage.data();

        // For each row of A
        for i in 0..m {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            
            // For each non-zero in this row of A
            for idx in row_start..row_end {
                let a_val = data[idx].clone();
                let col_a = indices[idx]; // This is the column index in A, which is row index in B
                
                // Multiply with each column of B
                for j in 0..n {
                    // B is row-major: B[col_a, j] = dense_data[col_a * n + j]
                    // C[i, j] += A[i, col_a] * B[col_a, j]
                    let b_val = dense_data[col_a * n + j].clone();
                    result[i * n + j] += a_val.clone() * b_val;
                }
            }
        }

        let result_storage = DenseStorage::from_vec(result, &[m, n])?;
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
    /// Dense tensor containing the sum
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn add_dense(
        &self,
        other: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone + Default + Send + Sync + 'static,
        T: num_traits::Zero + Clone + Copy + Default + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + core::ops::Neg<Output = T> + 'static,
    {
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "add_dense",
            });
        }

        // Convert sparse to dense at storage level then add
        let dense_storage = self.storage.to_dense().map_err(TensorError::StorageError)?;
        let dense_self = Tensor::<B, DenseStorage<T>, T>::from_storage(dense_storage, self.backend.clone());
        crate::ops::arithmetic::add(&dense_self, other)
    }

    /// Performs element-wise multiplication with a dense tensor.
    ///
    /// Multiplies each element of the dense tensor. The result preserves the sparse structure.
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
        T: num_traits::Zero + Clone + core::ops::Mul<Output = T>,
    {
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "mul_dense",
            });
        }

        let mut new_data = Vec::with_capacity(self.nnz());
        let data = self.as_slice();
        let indices = self.storage.indices();
        let indptr = self.storage.indptr();
        let dense_slice = other.as_slice();
        let cols = self.shape().dims()[1];

        for row in 0..self.shape().dims()[0] {
            let start = indptr[row];
            let end = indptr[row + 1];

            for i in start..end {
                let col = indices[i];
                let idx = row * cols + col;
                let val = data[i];
                let dense_val = dense_slice[idx];
                new_data.push(val * dense_val);
            }
        }

        // Create new CSR storage with same structure but new data
        let new_storage = crate::CsrStorage::new(
            new_data,
            indices.to_vec(),
            indptr.to_vec(),
            self.shape().dims(),
        );

        let new_storage = new_storage.map_err(TensorError::StorageError)?;
        Ok(Tensor::from_storage(new_storage, self.backend.clone()))
    }

    /// Performs element-wise addition with another sparse tensor.
    ///
    /// Adds corresponding elements of two sparse tensors. The result is generally
    /// denser than the inputs due to overlapping sparsity patterns.
    ///
    /// # Arguments
    /// * `other` - Sparse tensor to add
    ///
    /// # Returns
    /// Dense tensor (simplest return type for now, as sparsity pattern changes are complex to predict)
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn add_sparse(
        &self,
        other: &Tensor<B, crate::CsrStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Clone + Default + Send + Sync + 'static,
        T: num_traits::Zero + Clone + Default + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + core::ops::Neg<Output = T> + Copy + 'static,
    {
        // Convert both sparse tensors to dense at storage level
        let d1_storage = self.storage.to_dense().map_err(TensorError::StorageError)?;
        let d2_storage = other.storage.to_dense().map_err(TensorError::StorageError)?;
        
        let d1 = Tensor::<B, DenseStorage<T>, T>::from_storage(d1_storage, self.backend.clone());
        let d2 = Tensor::<B, DenseStorage<T>, T>::from_storage(d2_storage, other.backend.clone());
        
        crate::ops::arithmetic::add(&d1, &d2)
    }

    /// Performs element-wise multiplication with another sparse tensor.
    ///
    /// Multiplies corresponding elements of two sparse tensors. The result preserves
    /// the union of sparsity patterns from both inputs.
    ///
    /// # Arguments
    /// * `other` - Sparse tensor to multiply
    ///
    /// # Returns
    /// Sparse tensor in COO format (flexible for result)
    ///
    /// # Errors
    /// Returns error if shapes don't match or operation fails.
    pub fn mul_sparse(
        &self,
        other: &Tensor<B, crate::CsrStorage<T>, T>,
    ) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: num_traits::Zero + Clone + core::ops::Mul<Output = T> + PartialEq,
    {
        if self.shape().dims() != other.shape().dims() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                actual: other.shape().dims().to_vec(),
                operation: "mul_sparse",
            });
        }

        // For element-wise multiplication, result is non-zero only where BOTH are non-zero.
        // Intersection of sparsity patterns.

        let mut result_data = Vec::new();
        let mut result_rows = Vec::new();
        let mut result_cols = Vec::new();

        let data1 = self.as_slice();
        let indices1 = self.storage.indices();
        let indptr1 = self.storage.indptr();

        let data2 = other.as_slice();
        let indices2 = other.storage.indices();
        let indptr2 = other.storage.indptr();

        for row in 0..self.shape().dims()[0] {
            let start1 = indptr1[row];
            let end1 = indptr1[row + 1];

            let start2 = indptr2[row];
            let end2 = indptr2[row + 1];

            // Allow merging sorted indices
            let mut i = start1;
            let mut j = start2;

            while i < end1 && j < end2 {
                let col1 = indices1[i];
                let col2 = indices2[j];

                if col1 == col2 {
                    // Match found
                    let val = data1[i] * data2[j];
                    if val != T::zero() {
                        result_data.push(val);
                        result_rows.push(row);
                        result_cols.push(col1);
                    }
                    i += 1;
                    j += 1;
                } else if col1 < col2 {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }

        // Create CooStorage directly and construct tensor
        let coo_storage = crate::CooStorage::new(
            result_data,
            result_rows,
            result_cols,
            self.shape().dims(),
        ).map_err(TensorError::StorageError)?;
        
        Ok(Tensor::<B, crate::CooStorage<T>, T>::from_storage(coo_storage, self.backend.clone()))
    }

    // Note: transpose() is defined in ops/sparse/mod.rs to avoid duplication

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
    pub fn to_sparse(
        &self,
        format: &crate::SparseFormat,
    ) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone + Default,
    {
        match format {
            crate::SparseFormat::Csr | crate::SparseFormat::Csc | crate::SparseFormat::Coo => {
                // Convert to COO via storage layer
                let coo_storage = self.storage.to_coo()
                    .map_err(crate::TensorError::StorageError)?;
                Ok(Tensor::<B, crate::CooStorage<T>, T>::from_storage(
                    coo_storage,
                    self.backend.clone(),
                ))
            }
        }
    }
}
