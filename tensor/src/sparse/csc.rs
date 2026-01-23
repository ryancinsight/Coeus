use crate::{Backend, DataType, Result, Tensor};

// CSC (Compressed Sparse Column) implementations
impl<B, T> Tensor<B, crate::CscStorage<T>, T>
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
        crate::SparseFormat::Csc
    }

    // Note: to_dense() and transpose() are defined in ops/sparse/mod.rs to avoid duplication

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
    pub fn to_sparse(
        &self,
        format: &crate::SparseFormat,
    ) -> Result<Tensor<B, crate::CooStorage<T>, T>>
    where
        B: Clone,
        T: Clone + Default + num_traits::Zero,
    {
        // Convert to COO via storage layer - all formats route through COO
        let coo_storage = self.storage.to_coo().map_err(crate::TensorError::StorageError)?;
        
        match format {
            crate::SparseFormat::Csc | crate::SparseFormat::Csr | crate::SparseFormat::Coo => {
                // Return as COO tensor
                Ok(Tensor::<B, crate::CooStorage<T>, T>::from_storage(
                    coo_storage,
                    self.backend.clone(),
                ))
            }
        }
    }
}

