//! # Advanced Tensor Indexing
//!
//! Advanced indexing operations including boolean masking and fancy indexing.
//!
//! ## Boolean Indexing
//!
//! Boolean indexing allows selecting elements based on a boolean mask:
//!
//! ```rust
//! use coeus_tensor::Tensor;
//! use coeus_backend::CpuBackend;
//! use coeus_storage::DenseStorage;
//! use coeus_dtype::float::Float32;
//!
//! let data = vec![
//!     Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),
//!     Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)
//! ];
//! let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
//!     data, &[2, 3]
//! ).unwrap();
//!
//! // Create boolean mask (slice of bools)
//! let mask = vec![false, false, false, true, true, true];
//!
//! // Select elements where mask is true
//! let selected = tensor.boolean_index(&mask).unwrap();
//! // Result: [4.0, 5.0, 6.0]
//! ```
//!
//! ## Fancy Indexing
//!
//! Fancy indexing uses integer arrays to select arbitrary elements:
//!
//! ```rust
//! # use coeus_tensor::Tensor;
//! # use coeus_backend::CpuBackend;
//! # use coeus_storage::DenseStorage;
//! # use coeus_dtype::float::Float32;
//! # let data = vec![Float32::new(10.0), Float32::new(20.0), Float32::new(30.0), Float32::new(40.0), Float32::new(50.0)];
//! # let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[5]).unwrap();
//!
//! // Select elements at indices [0, 2, 4, 1]
//! let indices = [0i32, 2, 4, 1];
//! let selected = tensor.fancy_index(&indices).unwrap();
//! // Result: [10.0, 30.0, 50.0, 20.0]
//! ```
//!
//! ## Advanced Slicing
//!
//! Advanced slicing with start, end, and step parameters for each dimension:
//!
//! ```rust
//! # use coeus_tensor::Tensor;
//! # use coeus_backend::CpuBackend;
//! # use coeus_storage::DenseStorage;
//! # use coeus_dtype::float::Float32;
//! # let data = vec![Float32::new(0.0), Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0), Float32::new(5.0)];
//! # let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[6]).unwrap();
//!
//! // Slice with step: [1:5:2] -> elements at indices 1, 3
//! let slices = [(Some(1), Some(5), 2)];
//! let sliced = tensor.advanced_slice(&slices).unwrap();
//! // Result: [1.0, 3.0]
//!
//! // Reverse slice: [4:0:-1] -> elements at indices 4, 3, 2, 1
//! let reverse_slices = [(Some(4), Some(0), -1)];
//! let reversed = tensor.advanced_slice(&reverse_slices).unwrap();
//! // Result: [4.0, 3.0, 2.0, 1.0]
//! ```

use alloc::vec::Vec;

use crate::{error::TensorError, Backend, DataType, Tensor};

/// Boolean indexing operations for tensors with dense storage
impl<B, T> Tensor<B, coeus_storage::DenseStorage<T>, T>
where
    B: Backend + Default,
    T: DataType,
{
    /// Boolean indexing: select elements where mask is true
    ///
    /// # Arguments
    /// * `mask` - Slice of boolean values with same length as tensor
    ///
    /// # Returns
    /// 1D tensor containing elements where mask is true
    ///
    /// # Errors
    /// Returns `TensorError::ShapeMismatch` if mask length doesn't match tensor length
    pub fn boolean_index(&self, mask: &[bool]) -> Result<Self, TensorError> {
        let tensor_len = self.len();

        // Validate mask length matches tensor length
        if mask.len() != tensor_len {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![tensor_len],
                actual: alloc::vec![mask.len()],
                operation: "boolean_index",
            });
        }

        // Collect elements where mask is true
        let mut selected_elements = Vec::new();
        let self_slice = self.as_slice();

        for (i, &mask_val) in mask.iter().enumerate() {
            if mask_val {
                selected_elements.push(self_slice[i]);
            }
        }

        // Create result tensor - 1D with length equal to number of true elements
        let len = selected_elements.len();
        Self::from_vec(selected_elements, &[len])
    }

    /// Boolean indexing assignment: set elements where mask is true
    ///
    /// # Arguments
    /// * `mask` - Slice of boolean values with same length as tensor
    /// * `value` - Scalar value to assign to masked elements
    ///
    /// # Errors
    /// Returns `TensorError::ShapeMismatch` if mask length doesn't match tensor length
    pub fn boolean_assign(&mut self, mask: &[bool], value: T) -> Result<(), TensorError> {
        let tensor_len = self.len();

        // Validate mask length matches tensor length
        if mask.len() != tensor_len {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![tensor_len],
                actual: alloc::vec![mask.len()],
                operation: "boolean_assign",
            });
        }

        let self_slice = self.as_mut_slice();

        for (i, &mask_val) in mask.iter().enumerate() {
            if mask_val {
                self_slice[i] = value;
            }
        }

        Ok(())
    }

    /// Fancy indexing: select elements using integer index array
    ///
    /// # Arguments
    /// * `indices` - 1D array of integer indices to select
    ///
    /// # Returns
    /// New tensor containing elements at specified indices
    ///
    /// # Errors
    /// Returns `TensorError::IndexOutOfBounds` if any index is out of bounds
    #[allow(clippy::cast_sign_loss)]
    pub fn fancy_index(&self, indices: &[i32]) -> Result<Self, TensorError> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let tensor_len = self.len() as i32;

        // Validate indices are in bounds
        for &idx in indices {
            if idx < 0 || idx >= tensor_len {
                return Err(TensorError::ShapeMismatch {
                    #[allow(clippy::cast_sign_loss)]
                    expected: alloc::vec![0, tensor_len as usize],
                    #[allow(clippy::cast_sign_loss)]
                    actual: alloc::vec![idx as usize],
                    operation: "fancy_index",
                });
            }
        }

        // Collect selected elements
        let mut selected_elements = Vec::with_capacity(indices.len());
        let self_slice = self.as_slice();

        for &idx in indices {
            selected_elements.push(self_slice[idx as usize]);
        }

        // Create result tensor - 1D with length equal to number of indices
        Self::from_vec(selected_elements, &[indices.len()])
    }

    /// Fancy indexing assignment: set elements at specified indices
    ///
    /// # Arguments
    /// * `indices` - Array of integer indices to modify
    /// * `values` - Values to assign (must match indices length or be scalar)
    ///
    /// # Errors
    /// Returns `TensorError::IndexOutOfBounds` if any index is out of bounds
    /// or `TensorError::ShapeMismatch` if values shape is incompatible
    #[allow(clippy::cast_sign_loss)]
    pub fn fancy_assign(&mut self, indices: &[i32], values: &[T]) -> Result<(), TensorError> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let tensor_len = self.len() as i32;

        // Validate indices are in bounds
        for &idx in indices {
            if idx < 0 || idx >= tensor_len {
                return Err(TensorError::ShapeMismatch {
                    #[allow(clippy::cast_sign_loss)]
                    expected: alloc::vec![0, tensor_len as usize],
                    #[allow(clippy::cast_sign_loss)]
                    actual: alloc::vec![idx as usize],
                    operation: "fancy_assign",
                });
            }
        }

        // Validate values length matches indices length
        if values.len() != indices.len() {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![indices.len()],
                actual: alloc::vec![values.len()],
                operation: "fancy_assign",
            });
        }

        let self_slice = self.as_mut_slice();

        for (i, &idx) in indices.iter().enumerate() {
            self_slice[idx as usize] = values[i];
        }

        Ok(())
    }

    /// Advanced slicing with step parameters for each dimension
    ///
    /// # Arguments
    /// * `slices` - Slice specifications for each dimension as (start, end, step)
    ///   Use None for start/end to indicate default bounds
    ///
    /// # Returns
    /// New tensor with sliced elements
    ///
    /// # Errors
    /// Returns `TensorError::ShapeMismatch` if slice dimensions don't match tensor dimensions
    #[allow(clippy::cast_sign_loss)]
    pub fn advanced_slice(
        &self,
        slices: &[(Option<i32>, Option<i32>, i32)],
    ) -> Result<Self, TensorError> {
        let tensor_dims = self.shape().dims();

        // Validate slice dimensions match tensor dimensions
        if slices.len() != tensor_dims.len() {
            return Err(TensorError::ShapeMismatch {
                expected: alloc::vec![tensor_dims.len()],
                actual: alloc::vec![slices.len()],
                operation: "advanced_slice",
            });
        }

        // Calculate output shape and collect indices for each dimension
        let mut output_shape = Vec::new();
        let mut dim_indices = Vec::new();

        for (dim_idx, &(start_opt, end_opt, step)) in slices.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let dim_size = tensor_dims[dim_idx] as i32;

            // Resolve start and end bounds
            let start = start_opt.unwrap_or(if step >= 0 { 0 } else { dim_size - 1 });
            let end = end_opt.unwrap_or(if step >= 0 { dim_size } else { -1 });

            // Handle negative indices
            let start_idx = if start < 0 {
                (dim_size + start).max(0)
            } else {
                start.min(dim_size)
            };
            let end_idx = if end < 0 {
                (dim_size + end).max(-1)
            } else {
                end.min(dim_size)
            };

            // Generate indices for this dimension
            let mut indices = Vec::new();
            if step > 0 {
                let mut idx = start_idx;
                while idx < end_idx {
                    if idx >= 0 && idx < dim_size {
                        indices.push(idx);
                    }
                    idx += step;
                }
            } else if step < 0 {
                let mut idx = start_idx;
                while idx > end_idx {
                    if idx >= 0 && idx < dim_size {
                        indices.push(idx);
                    }
                    idx += step;
                }
            }

            output_shape.push(indices.len());
            dim_indices.push(indices);
        }

        // Generate all combinations of multi-dimensional indices
        let mut flat_indices = Vec::new();
        generate_multi_dim_indices(
            &dim_indices,
            &mut Vec::new(),
            &mut flat_indices,
            tensor_dims,
        );

        // Collect elements at the calculated indices
        let mut result_data = Vec::with_capacity(flat_indices.len());
        let self_slice = self.as_slice();

        for flat_idx in flat_indices {
            result_data.push(self_slice[flat_idx]);
        }

        Self::from_vec(result_data, &output_shape)
    }
}

/// Helper function to generate all combinations of multi-dimensional indices
#[allow(clippy::cast_sign_loss)]
fn generate_multi_dim_indices(
    dim_indices: &[Vec<i32>],
    current: &mut Vec<i32>,
    result: &mut Vec<usize>,
    tensor_dims: &[usize],
) {
    if current.len() == dim_indices.len() {
        // Convert multi-dimensional index to flat index
        let mut flat_idx = 0;
        let mut stride = 1;
        for (i, &idx) in current.iter().enumerate().rev() {
            flat_idx += idx as usize * stride;
            stride *= tensor_dims[i];
        }
        result.push(flat_idx);
        return;
    }

    let dim = current.len();
    for &idx in &dim_indices[dim] {
        current.push(idx);
        generate_multi_dim_indices(dim_indices, current, result, tensor_dims);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use alloc::vec;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_boolean_index_basic() {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[2, 3],
        )
        .unwrap();

        // Create mask for elements > 3 (last 3 elements)
        let mask = vec![false, false, false, true, true, true];

        let result = tensor.boolean_index(&mask).unwrap();

        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.as_slice()[0].get(), 4.0);
        assert_eq!(result.as_slice()[1].get(), 5.0);
        assert_eq!(result.as_slice()[2].get(), 6.0);
    }

    #[test]
    fn test_boolean_index_empty() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3],
        )
        .unwrap();

        // All false mask
        let mask = vec![false, false, false];

        let result = tensor.boolean_index(&mask).unwrap();

        assert_eq!(result.shape().dims(), &[0]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_boolean_index_length_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3],
        )
        .unwrap();

        let mask = vec![true, false]; // Wrong length

        assert!(tensor.boolean_index(&mask).is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_boolean_assign() {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let mut tensor =
            Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
                data,
                &[4],
            )
            .unwrap();

        // Set elements at indices 1 and 3 to 99.0
        let mask = vec![false, true, false, true];
        tensor.boolean_assign(&mask, Float32::new(99.0)).unwrap();

        let result = tensor.as_slice();
        assert_eq!(result[0].get(), 1.0); // unchanged
        assert_eq!(result[1].get(), 99.0); // changed
        assert_eq!(result[2].get(), 3.0); // unchanged
        assert_eq!(result[3].get(), 99.0); // changed
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_fancy_index_basic() {
        let data = vec![
            Float32::new(10.0),
            Float32::new(20.0),
            Float32::new(30.0),
            Float32::new(40.0),
            Float32::new(50.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[5],
        )
        .unwrap();

        // Select elements at indices [0, 2, 4, 1]
        let indices = [0i32, 2, 4, 1];
        let result = tensor.fancy_index(&indices).unwrap();

        assert_eq!(result.shape().dims(), &[4]);
        assert_eq!(result.as_slice()[0].get(), 10.0);
        assert_eq!(result.as_slice()[1].get(), 30.0);
        assert_eq!(result.as_slice()[2].get(), 50.0);
        assert_eq!(result.as_slice()[3].get(), 20.0);
    }

    #[test]
    fn test_fancy_index_out_of_bounds() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3],
        )
        .unwrap();

        // Index 5 is out of bounds for tensor of length 3
        let indices = [0i32, 5];
        assert!(tensor.fancy_index(&indices).is_err());
    }

    #[test]
    fn test_fancy_index_negative() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3],
        )
        .unwrap();

        // Negative indices should be out of bounds
        let indices = [-1i32];
        assert!(tensor.fancy_index(&indices).is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_fancy_assign() {
        let data = vec![
            Float32::new(10.0),
            Float32::new(20.0),
            Float32::new(30.0),
            Float32::new(40.0),
            Float32::new(50.0),
        ];
        let mut tensor =
            Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
                data,
                &[5],
            )
            .unwrap();

        // Set elements at indices [1, 3] to [100.0, 200.0]
        let indices = [1i32, 3];
        let values = [Float32::new(100.0), Float32::new(200.0)];
        tensor.fancy_assign(&indices, &values).unwrap();

        let result = tensor.as_slice();
        assert_eq!(result[0].get(), 10.0); // unchanged
        assert_eq!(result[1].get(), 100.0); // changed
        assert_eq!(result[2].get(), 30.0); // unchanged
        assert_eq!(result[3].get(), 200.0); // changed
        assert_eq!(result[4].get(), 50.0); // unchanged
    }

    #[test]
    fn test_fancy_assign_length_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let mut tensor =
            Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
                data,
                &[3],
            )
            .unwrap();

        let indices = [0i32, 1];
        let values = [Float32::new(10.0)]; // Wrong length
        assert!(tensor.fancy_assign(&indices, &values).is_err());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_advanced_slice_1d_basic() {
        let data = vec![
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[6],
        )
        .unwrap();

        // Slice [1:5:2] -> elements at indices 1, 3
        let slices = [(Some(1), Some(5), 2)];
        let result = tensor.advanced_slice(&slices).unwrap();

        assert_eq!(result.shape().dims(), &[2]);
        assert_eq!(result.as_slice()[0].get(), 1.0);
        assert_eq!(result.as_slice()[1].get(), 3.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_advanced_slice_1d_reverse() {
        let data = vec![
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[6],
        )
        .unwrap();

        // Slice [4:0:-1] -> elements at indices 4, 3, 2, 1
        let slices = [(Some(4), Some(0), -1)];
        let result = tensor.advanced_slice(&slices).unwrap();

        assert_eq!(result.shape().dims(), &[4]);
        assert_eq!(result.as_slice()[0].get(), 4.0);
        assert_eq!(result.as_slice()[1].get(), 3.0);
        assert_eq!(result.as_slice()[2].get(), 2.0);
        assert_eq!(result.as_slice()[3].get(), 1.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_advanced_slice_2d() {
        let data = vec![
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
            Float32::new(10.0),
            Float32::new(11.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3, 4],
        )
        .unwrap();

        // Slice rows [0:2:1], cols [1:4:2] -> elements at (0,1), (0,3), (1,1), (1,3)
        let slices = [(Some(0), Some(2), 1), (Some(1), Some(4), 2)];
        let result = tensor.advanced_slice(&slices).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.as_slice()[0].get(), 1.0); // (0,1)
        assert_eq!(result.as_slice()[1].get(), 3.0); // (0,3)
        assert_eq!(result.as_slice()[2].get(), 5.0); // (1,1)
        assert_eq!(result.as_slice()[3].get(), 7.0); // (1,3)
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_advanced_slice_defaults() {
        let data = vec![
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[6],
        )
        .unwrap();

        // Slice [::2] -> all elements with step 2
        let slices = [(None, None, 2)];
        let result = tensor.advanced_slice(&slices).unwrap();

        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.as_slice()[0].get(), 0.0);
        assert_eq!(result.as_slice()[1].get(), 2.0);
        assert_eq!(result.as_slice()[2].get(), 4.0);
    }

    #[test]
    fn test_advanced_slice_dimension_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = Tensor::<CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>::from_vec(
            data,
            &[3],
        )
        .unwrap();

        // Wrong number of dimensions
        let slices = [(Some(0), Some(2), 1), (Some(0), Some(2), 1)];
        assert!(tensor.advanced_slice(&slices).is_err());
    }
}

