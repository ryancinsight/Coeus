//! Tensor shape manipulation operations.
//!
//! This module provides operations for reshaping and transposing tensors.

use std::{format, string::ToString, vec, vec::Vec};

/// Shape manipulation operations for tensors with dense storage.
///
/// This trait provides methods for reshaping and transposing tensors.
///
impl<B, T> crate::Tensor<B, coeus_storage::DenseStorage<T>, T>
where
    B: crate::Backend + Default,
    T: crate::DataType,
{

    /// Transposes dimensions of the tensor.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to transpose
    /// * `dim1` - Second dimension to transpose
    ///
    /// # Errors
    /// * Returns `TensorError::ShapeError` if either dimension is out of bounds
    /// * Returns `TensorError::InvalidOperation` for tensors with more than 2 dimensions
    ///
    /// # Examples
    /// ```
    /// use coeus_tensor::Tensor;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
    /// let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
    ///     data,
    ///     &[2, 2]
    /// ).unwrap();
    ///
    /// // Transpose dimensions 0 and 1
    /// let transposed = tensor.transpose(0, 1).unwrap();
    /// assert_eq!(transposed.shape().dims(), &[2, 2]);
    /// // Data layout: [1,3,2,4]
    /// ```
    pub fn transpose(&self, dim0: usize, dim1: usize) -> crate::Result<Self> {
        // Validate dimensions
        let ndim = self.shape().dims().len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(crate::TensorError::ShapeError {
                expected: ndim,
                actual: dim0.max(dim1),
                message: format!("Dimension out of bounds: dim0={dim0}, dim1={dim1}, ndim={ndim}"),
            });
        }

        // If transposing the same dimension, return a copy (identity operation)
        if dim0 == dim1 {
            let data = self.as_slice().to_vec();
            let new_storage = coeus_storage::DenseStorage::from_vec(data, self.shape().dims())
                .map_err(crate::TensorError::StorageError)?;
            return Ok(Self::from_storage(new_storage, B::default()));
        }

        // For 2D transpose, we can do a simple reordering
        if ndim == 2 {
            let rows = self.shape().dims()[0];
            let cols = self.shape().dims()[1];
            let mut transposed_data = Vec::with_capacity(self.len());

            // Transpose: (i,j) -> (j,i)
            for j in 0..cols {
                for i in 0..rows {
                    let linear_idx = i * cols + j;
                    transposed_data.push(self.as_slice()[linear_idx]);
                }
            }

            let new_dims = vec![cols, rows];
            let new_storage = coeus_storage::DenseStorage::from_vec(transposed_data, &new_dims)
                .map_err(crate::TensorError::StorageError)?;

            Ok(Self::from_storage(new_storage, B::default()))
        } else {
            // For higher dimensions, use a more general approach
            // This is a simplified implementation - full zero-copy transpose would require stride manipulation
            Err(crate::TensorError::ShapeError {
                expected: 2,
                actual: ndim,
                message: format!(
                    "Transpose for {ndim}-D tensors not yet implemented. Only 2D tensors are supported. Use reshape to convert to 2D first."
                ),
            })
        }
    }

    /// Helper method to resolve reshape dimensions with -1 inference.
    fn resolve_reshape_dims(&self, dims: &[isize]) -> crate::Result<Vec<usize>> {
        let mut result = Vec::with_capacity(dims.len());
        let mut infer_idx = None;

        // First pass: collect known dimensions and find inference point
        let mut known_product = 1usize;
        for (i, &dim) in dims.iter().enumerate() {
            if dim == -1 {
                if infer_idx.is_some() {
                    return Err(crate::TensorError::ShapeError {
                        expected: 0,
                        actual: 0,
                        message: "Multiple -1 dimensions in reshape".to_string(),
                    });
                }
                infer_idx = Some(i);
                result.push(0); // Placeholder
            } else if dim <= 0 {
                return Err(crate::TensorError::ShapeError {
                    expected: 0,
                    actual: 0,
                    message: format!("Invalid dimension {dim} in reshape"),
                });
            } else {
                let dim_usize =
                    usize::try_from(dim).map_err(|_| crate::TensorError::ShapeError {
                        expected: 0,
                        actual: 0,
                        message: format!("Dimension overflow in reshape: {dim} exceeds usize::MAX"),
                    })?;
                result.push(dim_usize);
                known_product = known_product.checked_mul(dim_usize).ok_or_else(|| {
                    crate::TensorError::ShapeError {
                        expected: 0,
                        actual: 0,
                        message: "Dimension product overflow in reshape".to_string(),
                    }
                })?;
            }
        }

        // Infer the -1 dimension
        if let Some(idx) = infer_idx {
            let total_elements = self.len();
            if total_elements % known_product != 0 {
                return Err(crate::TensorError::ShapeError {
                    expected: known_product,
                    actual: total_elements,
                    message: "Cannot infer -1 dimension: total elements not divisible by known dimensions".to_string(),
                });
            }
            result[idx] = total_elements / known_product;
        }

        Ok(result)
    }
}

/// Generic reshape operations for any storage type.
///
/// This converts tensors to dense storage for reshaping operations.
impl<B, S, T> crate::Tensor<B, S, T>
where
    B: crate::Backend + Default + Clone,
    S: crate::Storage<T> + crate::StorageFromVec<T> + crate::StorageToDense<T> + 'static,
    T: crate::DataType + Clone,
{
    /// Reshapes the tensor to new dimensions.
    ///
    /// This method converts the tensor to dense storage if needed, then reshapes it.
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    /// * `dims` - New dimensions. Use -1 to infer one dimension.
    ///
    /// # Returns
    /// A new tensor with the reshaped dimensions (always dense storage).
    ///
    /// # Errors
    /// Returns error if total element count doesn't match or conversion fails.
    pub fn reshape(&self, dims: &[isize]) -> crate::Result<crate::Tensor<B, crate::DenseStorage<T>, T>> {
        // Convert to dense storage for arbitrary element rearrangement
        let dense_tensor = self.to_dense_generic()?;

        // Validate and resolve dimensions
        let resolved_dims = crate::Tensor::<B, S, T>::resolve_reshape_dims_generic(dense_tensor.len(), dims)?;

        // Check total element count matches
        let new_size: usize = resolved_dims.iter().product();
        if new_size != dense_tensor.len() {
            return Err(crate::TensorError::ShapeError {
                expected: dense_tensor.len(),
                actual: new_size,
                message: "Total element count mismatch in reshape".to_string(),
            });
        }

        // Create new dense storage with reshaped data
        let data = dense_tensor.as_slice().to_vec();
        let new_storage = crate::DenseStorage::from_vec(data, &resolved_dims)
            .map_err(crate::TensorError::StorageError)?;

        Ok(crate::Tensor::from_storage(new_storage, dense_tensor.backend))
    }
}
