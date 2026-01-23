use crate::{Backend, DataType, Result, Tensor};

// COO (Coordinate) implementations
impl<B, T> Tensor<B, crate::CooStorage<T>, T>
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
        crate::SparseFormat::Coo
    }

    /// Sorts the COO tensor by row, then by column for efficient operations.
    pub fn sort(&mut self) {
        self.storage.sort();
    }

    // Note: to_dense() and transpose() are defined in ops/sparse/mod.rs to avoid duplication

    /// Computes the sum of all non-zero elements in the sparse tensor.
    #[must_use]
    pub fn sum(&self) -> T
    where
        T: Default + core::ops::Add<Output = T>,
    {
        self.storage.data()
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
            let sum: f64 = self.storage.data().iter().map(|x| (*x).into()).sum();
            sum / nnz
        }
    }

    /// Converts COO sparse tensor to different sparse formats.
    ///
    /// # Note
    /// Currently always returns COO since that's the return type.
    /// The operation performs format conversion through the storage layer.
    pub fn to_sparse(
        &self,
        format: &crate::SparseFormat,
    ) -> Result<Self>
    where
        B: Clone,
        T: Clone + Default + num_traits::Zero,
        Self: Sized,
    {
        match format {
            crate::SparseFormat::Coo => {
                // Already COO - clone and return
                Ok(Self::from_storage(self.storage.clone(), self.backend.clone()))
            }
            crate::SparseFormat::Csr | crate::SparseFormat::Csc => {
                // Convert via storage layer, then back to COO
                let csr = self.storage.to_csr().map_err(crate::TensorError::StorageError)?;
                let new_coo = csr.to_coo().map_err(crate::TensorError::StorageError)?;
                Ok(Self::from_storage(new_coo, self.backend.clone()))
            }
        }
    }
}

