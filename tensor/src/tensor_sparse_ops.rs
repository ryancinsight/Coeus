impl<B, T> Tensor<B, CsrStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Creates a CSR sparse tensor from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values in row-major order
    /// * `indices` - Column indices for each non-zero element
    /// * `indptr` - Row pointers (must have shape[0] + 1 elements)
    /// * `shape` - Matrix shape [rows, cols]
    /// * `backend` - Backend instance
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::CsrStorage;
    /// use dtype::float::Float32;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = CpuBackend::new();
    /// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    /// let indices = vec![0, 2, 1];
    /// let indptr = vec![0, 2, 3];
    /// let tensor = Tensor::<CpuBackend, CsrStorage<Float32>, Float32>::from_csr(
    ///     data, indices, indptr, &[2, 3], backend
    /// )?;
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_csr(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
        backend: B,
    ) -> Result<Self> {
        let storage = CsrStorage::new(data, indices, indptr, shape)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Performs sparse-sparse matrix multiplication.
    ///
    /// Computes C = A @ B where both A and B are sparse tensors in CSR format.
    /// Uses efficient sparse-sparse multiplication algorithms with O(nnz_A + nnz_B) complexity.
    ///
    /// # Arguments
    /// * `other` - Right-hand side sparse tensor for multiplication
    ///
    /// # Returns
    /// Result tensor containing the matrix product in CSR format
    ///
    /// # Errors
    /// Returns error if tensor shapes are incompatible
    #[tracing::instrument(level = "trace", skip(self, other), fields(lhs_shape = ?self.shape().dims(), rhs_shape = ?other.shape().dims()))]
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        B: Clone + Default,
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        let lhs_shape = self.shape().dims();
        let rhs_shape = other.shape().dims();

        // Validate 2D matrices
        if lhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: lhs_shape.len(),
                message: format!("Left matrix must be 2D, got shape {lhs_shape:?}"),
            });
        }
        if rhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: rhs_shape.len(),
                message: format!("Right matrix must be 2D, got shape {rhs_shape:?}"),
            });
        }

        // Validate compatible dimensions
        let m = lhs_shape[0];
        let n = lhs_shape[1];
        let p = rhs_shape[1];

        if n != rhs_shape[0] {
            return Err(TensorError::ShapeError {
                expected: n,
                actual: rhs_shape[0],
                message: format!(
                    "Matrix dimension mismatch: {}�{} @ {}�{} (inner dimensions {} ? {})",
                    m, n, rhs_shape[0], p, n, rhs_shape[0]
                ),
            });
        }

        // Use sparse matrix multiplication
        let result_coo = self.storage.matmul_sparse(&other.storage, SparseFormat::Coo)?;
        let result_csr = result_coo.to_csr();

        Ok(Self::from_storage(result_csr, self.backend.clone()))
    }
}

impl<B, T> Tensor<B, CscStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Creates a CSC sparse tensor from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values in column-major order
    /// * `indices` - Row indices for each non-zero element
    /// * `indptr` - Column pointers (must have shape[1] + 1 elements)
    /// * `shape` - Matrix shape [rows, cols]
    /// * `backend` - Backend instance
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    pub fn from_csc(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: &[usize],
        backend: B,
    ) -> Result<Self> {
        let storage = CscStorage::new(data, indices, indptr, shape)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Performs sparse-sparse matrix multiplication.
    ///
    /// Computes C = A @ B where both A and B are sparse tensors in CSC format.
    /// Uses efficient sparse-sparse multiplication algorithms with O(nnz_A + nnz_B) complexity.
    ///
    /// # Arguments
    /// * `other` - Right-hand side sparse tensor for multiplication
    ///
    /// # Returns
    /// Result tensor containing the matrix product in CSC format
    ///
    /// # Errors
    /// Returns error if tensor shapes are incompatible
    #[tracing::instrument(level = "trace", skip(self, other), fields(lhs_shape = ?self.shape().dims(), rhs_shape = ?other.shape().dims()))]
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        B: Clone + Default,
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        let lhs_shape = self.shape().dims();
        let rhs_shape = other.shape().dims();

        // Validate 2D matrices
        if lhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: lhs_shape.len(),
                message: format!("Left matrix must be 2D, got shape {lhs_shape:?}"),
            });
        }
        if rhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: rhs_shape.len(),
                message: format!("Right matrix must be 2D, got shape {rhs_shape:?}"),
            });
        }

        // Validate compatible dimensions
        let m = lhs_shape[0];
        let n = lhs_shape[1];
        let p = rhs_shape[1];

        if n != rhs_shape[0] {
            return Err(TensorError::ShapeError {
                expected: n,
                actual: rhs_shape[0],
                message: format!(
                    "Matrix dimension mismatch: {}�{} @ {}�{} (inner dimensions {} ? {})",
                    m, n, rhs_shape[0], p, n, rhs_shape[0]
                ),
            });
        }

        // Use sparse matrix multiplication
        let result_coo = self.storage.matmul_sparse(&other.storage, SparseFormat::Coo)?;
        let result_csc = result_coo.to_csc();

        Ok(Self::from_storage(result_csc, self.backend.clone()))
    }
}

impl<B, T> Tensor<B, CooStorage<T>, T>
where
    B: Backend,
    T: DataType,
{
    /// Creates a COO sparse tensor from components.
    ///
    /// # Arguments
    /// * `data` - Non-zero values
    /// * `row_indices` - Row indices for each non-zero element
    /// * `col_indices` - Column indices for each non-zero element
    /// * `shape` - Matrix shape [rows, cols]
    /// * `backend` - Backend instance
    ///
    /// # Errors
    /// Returns error if dimensions are inconsistent or indices out of bounds.
    ///
    /// # Examples
    /// ```
    /// use tensor::Tensor;
    /// use backend::CpuBackend;
    /// use storage::CooStorage;
    /// use dtype::float::Float32;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = CpuBackend::new();
    /// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    /// let row_indices = vec![0, 0, 1];
    /// let col_indices = vec![0, 2, 1];
    /// let tensor = Tensor::<CpuBackend, CooStorage<Float32>, Float32>::from_coo(
    ///     data, row_indices, col_indices, &[2, 3], backend
    /// )?;
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_coo(
        data: Vec<T>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: &[usize],
        backend: B,
    ) -> Result<Self> {
        let storage = CooStorage::new(data, row_indices, col_indices, shape)?;
        Ok(Self::from_storage(storage, backend))
    }

    /// Performs sparse-sparse matrix multiplication.
    ///
    /// Computes C = A @ B where both A and B are sparse tensors in COO format.
    /// Uses efficient sparse-sparse multiplication algorithms with O(nnz_A + nnz_B) complexity.
    ///
    /// # Arguments
    /// * `other` - Right-hand side sparse tensor for multiplication
    ///
    /// # Returns
    /// Result tensor containing the matrix product in COO format
    ///
    /// # Errors
    /// Returns error if tensor shapes are incompatible
    #[tracing::instrument(level = "trace", skip(self, other), fields(lhs_shape = ?self.shape().dims(), rhs_shape = ?other.shape().dims()))]
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        B: Clone + Default,
        T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + num_traits::Zero + Copy,
    {
        let lhs_shape = self.shape().dims();
        let rhs_shape = other.shape().dims();

        // Validate 2D matrices
        if lhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: lhs_shape.len(),
                message: format!("Left matrix must be 2D, got shape {lhs_shape:?}"),
            });
        }
        if rhs_shape.len() != 2 {
            return Err(TensorError::ShapeError {
                expected: 2,
                actual: rhs_shape.len(),
                message: format!("Right matrix must be 2D, got shape {rhs_shape:?}"),
            });
        }

        // Validate compatible dimensions
        let m = lhs_shape[0];
        let n = lhs_shape[1];
        let p = rhs_shape[1];

        if n != rhs_shape[0] {
            return Err(TensorError::ShapeError {
                expected: n,
                actual: rhs_shape[0],
                message: format!(
                    "Matrix dimension mismatch: {}�{} @ {}�{} (inner dimensions {} ? {})",
                    m, n, rhs_shape[0], p, n, rhs_shape[0]
                ),
            });
        }

        // Use sparse matrix multiplication
        let result_coo = self.storage.matmul_sparse(&other.storage, SparseFormat::Coo)?;

        Ok(Self::from_storage(result_coo, self.backend.clone()))
    }
}

// Reduction operations

// Sparse tensor operations and conversions

/// Utility functions for sparse tensor operations

