//! Tensor creation operations.
//!
//! This module provides functions for creating tensors from various data sources
//! and generating tensors with specific fill patterns.

// Convenience operations
pub use tensor_creation_convenience::*;

#[allow(missing_docs)]
mod tensor_creation_convenience {
    use dtype;
    use rand::prelude::*;
    use std::sync::Mutex;
    use std::vec::Vec;

    /// Type alias for the most common CPU float32 tensor type.
    pub type CpuF32Tensor = crate::Tensor<
        crate::CpuBackend<dtype::float::Float32>,
        crate::DenseStorage<dtype::float::Float32>,
        dtype::float::Float32,
    >;

    /// Creates a tensor filled with random values from a normal distribution.
    ///
    /// This is a convenience method for CPU + DenseStorage + Float32 tensors.
    ///
    /// # Arguments
    /// * `shape` - Shape dimensions for the tensor
    ///
    /// # Returns
    /// A CPU Float32 tensor with normally distributed random values
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    ///
    /// # Note
    /// This method is a convenience function that uses CPU backend with Float32 data type.
    /// For full control over backend/storage types, use the generic constructor methods.
    pub fn randn(shape: &[usize]) -> crate::Result<CpuF32Tensor> {
        // Global RNG for deterministic random number generation
        static RNG: Mutex<Option<rand::rngs::StdRng>> = Mutex::new(None);

        let mut rng_lock = RNG.lock().unwrap();
        let rng = rng_lock.get_or_insert_with(rand::rngs::StdRng::from_entropy);

        let total_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total_elements);

        // Generate random values from standard normal distribution
        for _ in 0..total_elements {
            let value: f32 = rng.sample(rand::distributions::Standard);
            data.push(dtype::float::Float32::new(value));
        }

        CpuF32Tensor::from_vec(data, shape)
    }

    /// Concatenates tensors along a specified dimension.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate
    /// * `dim` - Dimension along which to concatenate
    ///
    /// # Returns
    /// A new tensor with the concatenated result
    ///
    /// # Errors
    /// Returns error if concatenation fails
    ///
    /// # Note
    /// All input tensors must have the same shape except for the concatenation dimension.
    pub fn cat(tensors: &[CpuF32Tensor], dim: usize) -> crate::Result<CpuF32Tensor> {
        if tensors.is_empty() {
            return Err(crate::TensorError::ShapeError {
                expected: 0,
                actual: 0,
                message: "Cannot concatenate empty tensor list".to_string(),
            });
        }

        // Check all tensors have compatible shapes
        let first_shape = tensors[0].shape().dims();
        if dim >= first_shape.len() {
            return Err(crate::TensorError::ShapeError {
                expected: 0,
                actual: dim,
                message: format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    dim,
                    first_shape.len()
                ),
            });
        }

        // Verify all tensors have compatible shapes (same size in all dimensions except dim)
        for (i, tensor) in tensors.iter().enumerate() {
            let shape = tensor.shape().dims();
            if shape.len() != first_shape.len() {
                return Err(crate::TensorError::ShapeError {
                    expected: first_shape.len(),
                    actual: shape.len(),
                    message: format!(
                        "Tensor {} has {} dimensions, expected {}",
                        i,
                        shape.len(),
                        first_shape.len()
                    ),
                });
            }

            for (j, (&actual, &expected)) in shape.iter().zip(first_shape).enumerate() {
                if j != dim && actual != expected {
                    return Err(crate::TensorError::ShapeError {
                        expected,
                        actual,
                        message: format!(
                            "Tensor {} dimension {} has size {}, expected {}",
                            i, j, actual, expected
                        ),
                    });
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        let total_dim_size: usize = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
        output_shape[dim] = total_dim_size;

        // Calculate total number of elements
        let total_elements: usize = output_shape.iter().product();

        // Concatenate the data
        let mut concatenated_data = vec![dtype::float::Float32::default(); total_elements];

        let mut offsets = vec![0; output_shape.len()];
        for tensor in tensors {
            let tensor_shape = tensor.shape().dims();
            let tensor_size = tensor_shape.iter().product::<usize>();

            // Copy this tensor's data with proper index calculation
            for linear_idx in 0..tensor_size {
                // Convert linear index to multi-dimensional coordinates
                let mut coords = vec![0; tensor_shape.len()];
                let mut remaining = linear_idx;
                for (i, &dim_size) in tensor_shape.iter().enumerate().rev() {
                    coords[i] = remaining % dim_size;
                    remaining /= dim_size;
                }

                // Apply offset for concatenation dimension
                coords[dim] += offsets[dim];

                // Convert back to linear index in output tensor
                let mut output_linear_idx = 0;
                let mut multiplier = 1;
                for (i, &coord) in coords.iter().enumerate().rev() {
                    output_linear_idx += coord * multiplier;
                    multiplier *= output_shape[i];
                }

                // Copy the element
                concatenated_data[output_linear_idx] = tensor.as_slice()[linear_idx];
            }

            // Update offset for next tensor
            offsets[dim] += tensor_shape[dim];
        }

        CpuF32Tensor::from_vec(concatenated_data, &output_shape)
    }
}
