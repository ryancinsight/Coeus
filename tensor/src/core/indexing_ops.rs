//! Indexing and advanced indexing operations for tensors
//!
//! This module contains operations for accessing and modifying tensor elements
//! using various indexing schemes including slicing, gathering, and scattering.

use crate::{Tensor, TensorError, Dtype, Result};
use crate::ops::indexing::Slice;

#[cfg(test)]
mod indexing_ops_tests {
    use super::*;

    /// Test basic slicing operation
    #[test]
    fn test_slice_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let slice = tensor.slice(&[Slice::Range(0..1), Slice::Range(0..2)]).unwrap();
        assert_eq!(slice.data(), &[1.0, 2.0]);
        assert_eq!(slice.shape(), &[1, 2]);
    }

    /// Test slice with step
    #[test]
    fn test_slice_with_step() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let slice = tensor.slice(&[Slice::Range(0..2), Slice::Range(0..3)]).unwrap();
        assert_eq!(slice.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(slice.shape(), &[2, 3]);
    }

    /// Test slice error handling
    #[test]
    fn test_slice_error_handling() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.slice(&[Slice::Index(0)]);
        assert!(result.is_err());
    }

    /// Test gather operation
    #[test]
    fn test_gather_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = Tensor::from_vec(vec![0, 1], vec![2]);
        let result = tensor.gather(0, &indices).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test gather with different dimension
    #[test]
    fn test_gather_dimension() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = Tensor::from_vec(vec![0, 1], vec![2]);
        let result = tensor.gather(1, &indices).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test scatter operation
    #[test]
    fn test_scatter_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let indices = Tensor::from_vec(vec![0, 1], vec![2]);
        let src = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.scatter(0, &indices, &src).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Test index_select operation
    #[test]
    fn test_index_select_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.index_select(0, &[0, 1]).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    /// Test advanced indexing
    #[test]
    fn test_advanced_index_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let indices = vec![&Tensor::from_vec(vec![0, 1], vec![2])];
        let result = tensor.advanced_index(&indices).unwrap();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Test index_put operation
    #[test]
    fn test_index_put_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let indices = vec![&Tensor::from_vec(vec![0, 1], vec![2])];
        let values = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let result = tensor.index_put(&indices, &values).unwrap();
        assert_eq!(result.data(), &[1.0, 0.0, 2.0, 0.0]);
    }

    /// Test index_add operation
    #[test]
    fn test_index_add_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = vec![&Tensor::from_vec(vec![0, 1], vec![2])];
        let values = Tensor::from_vec(vec![1.0, 1.0], vec![2, 1]);
        let result = tensor.index_add(&indices, &values).unwrap();
        assert_eq!(result.data(), &[2.0, 2.0, 4.0, 4.0]);
    }

    /// Test index_copy operation
    #[test]
    fn test_index_copy_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        let indices = vec![&Tensor::from_vec(vec![0, 1], vec![2])];
        let values = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let result = tensor.index_copy(&indices, &values).unwrap();
        assert_eq!(result.data(), &[1.0, 0.0, 2.0, 0.0]);
    }

    /// Test edge cases for indexing operations
    #[test]
    fn test_indexing_edge_cases() {
        // Test with empty indices
        let tensor = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let indices = Tensor::from_vec(vec![], vec![0]);
        let result = tensor.gather(0, &indices);
        assert!(result.is_err());

        // Test with out of bounds indices
        let indices_oob = Tensor::from_vec(vec![10], vec![1]);
        let result_oob = tensor.gather(0, &indices_oob);
        assert!(result_oob.is_err());
    }

    /// Test indexing operations preserve gradients
    #[test]
    fn test_indexing_gradient_preservation() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tensor.set_requires_grad(true);

        let indices = Tensor::from_vec(vec![0, 1], vec![2]);
        let result = tensor.gather(0, &indices).unwrap();

        assert!(result.requires_grad());
    }

    /// Test multidimensional indexing
    #[test]
    fn test_multidimensional_indexing() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let indices = vec![
            &Tensor::from_vec(vec![0, 1], vec![2]),
            &Tensor::from_vec(vec![0, 2], vec![2])
        ];
        let result = tensor.advanced_index(&indices).unwrap();
        assert_eq!(result.data(), &[1.0, 6.0]);
        assert_eq!(result.shape(), &[2]);
    }

    /// Test edge cases for all indexing operations
    #[test]
    fn test_indexing_edge_cases() {
        // Test with empty tensors
        let empty_tensor = Tensor::from_vec(vec![], vec![0]);
        let empty_indices = vec![&Tensor::from_vec(vec![], vec![0])];
        let empty_values = Tensor::from_vec(vec![], vec![0]);

        let scatter_result = empty_tensor.scatter(0, &empty_indices[0], &empty_values);
        assert!(scatter_result.is_ok());

        // Test with single element tensors
        let single_tensor = Tensor::from_vec(vec![42.0], vec![1]);
        let single_indices = vec![&Tensor::from_vec(vec![0], vec![1])];
        let single_values = Tensor::from_vec(vec![99.0], vec![1]);

        let scatter_single = single_tensor.scatter(0, &single_indices[0], &single_values).unwrap();
        assert_eq!(scatter_single.data(), &[99.0]);

        // Test with large tensors
        let large_data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let large_tensor = Tensor::from_vec(large_data, vec![1000]);
        let large_indices = Tensor::from_vec(vec![0, 1, 999], vec![3]);
        let large_values = Tensor::from_vec(vec![-1.0, -2.0, -3.0], vec![3]);

        let scatter_large = large_tensor.scatter(0, &large_indices, &large_values).unwrap();
        assert_eq!(scatter_large.data()[0], -1.0);
        assert_eq!(scatter_large.data()[1], -2.0);
        assert_eq!(scatter_large.data()[999], -3.0);
    }

    /// Test error conditions for indexing operations
    #[test]
    fn test_indexing_error_conditions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test invalid slice indices
        let invalid_slice = tensor.slice(&[Slice::Range(5..10)]);
        assert!(invalid_slice.is_err());

        // Test invalid gather indices
        let invalid_gather_indices = Tensor::from_vec(vec![-1, 10], vec![2]);
        let invalid_gather = tensor.gather(0, &invalid_gather_indices);
        assert!(invalid_gather.is_err());

        // Test mismatched dimensions for scatter
        let mismatched_values = Tensor::from_vec(vec![1.0], vec![1]);
        let indices = vec![&Tensor::from_vec(vec![0], vec![1])];
        let invalid_scatter = tensor.scatter(0, &indices[0], &mismatched_values);
        assert!(invalid_scatter.is_err());

        // Test out of bounds indices for index_select
        let out_of_bounds_indices = &[10, 20];
        let invalid_select = tensor.index_select(0, out_of_bounds_indices);
        assert!(invalid_select.is_err());
    }

    /// Test memory safety and bounds checking
    #[test]
    fn test_indexing_memory_safety() {
        // Test with very large indices
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let large_indices = Tensor::from_vec(vec![0, 1, 0, 1], vec![4]);

        // This should not cause memory corruption
        let gather_result = tensor.gather(0, &large_indices);
        assert!(gather_result.is_ok());

        // Test with repeated indices
        let repeated_indices = Tensor::from_vec(vec![0, 0, 1, 1], vec![4]);
        let repeated_gather = tensor.gather(1, &repeated_indices);
        assert!(repeated_gather.is_ok());

        // Test scatter with overlapping indices
        let overlapping_indices = Tensor::from_vec(vec![0, 0, 1], vec![3]);
        let overlapping_values = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let scatter_overlap = tensor.scatter(0, &overlapping_indices, &overlapping_values);
        assert!(scatter_overlap.is_ok());
    }

    /// Test performance characteristics
    #[test]
    fn test_indexing_performance() {
        // Test with medium-sized tensors
        let data: Vec<f64> = (0..10000).map(|x| x as f64).collect();
        let tensor = Tensor::from_vec(data, vec![100, 100]);

        let indices = Tensor::from_vec(vec![0, 50, 99], vec![3]);

        // These operations should complete in reasonable time
        let gather_result = tensor.gather(0, &indices).unwrap();
        assert_eq!(gather_result.shape(), &[3, 100]);

        let scatter_values = Tensor::from_vec(vec![-1.0, -2.0, -3.0], vec![3]);
        let scatter_result = tensor.scatter(0, &indices, &scatter_values).unwrap();
        assert_eq!(scatter_result.shape(), &[100, 100]);
    }

    /// Test numerical precision for indexing operations
    #[test]
    fn test_indexing_precision() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.14159, 2.71828], vec![2, 2]);
        let indices = Tensor::from_vec(vec![0, 1], vec![2]);

        let gather_result = tensor.gather(0, &indices).unwrap();
        assert_relative_eq!(gather_result.data()[0], 1.0, epsilon = 1e-15);
        assert_relative_eq!(gather_result.data()[1], 2.0, epsilon = 1e-15);
        assert_relative_eq!(gather_result.data()[2], 3.14159, epsilon = 1e-6);
        assert_relative_eq!(gather_result.data()[3], 2.71828, epsilon = 1e-6);
    }

    /// Test gradient computation for indexing operations
    #[test]
    fn test_indexing_gradients() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tensor.set_requires_grad(true);

        let indices = Tensor::from_vec(vec![0, 1], vec![2]);
        let gather_result = tensor.gather(0, &indices).unwrap();

        assert!(gather_result.requires_grad());

        // Test that gradients can be computed
        // (Actual gradient computation would require backward pass implementation)
    }

    /// Test complex indexing patterns
    #[test]
    fn test_complex_indexing_patterns() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Test multiple gather operations
        let indices1 = Tensor::from_vec(vec![0], vec![1]);
        let indices2 = Tensor::from_vec(vec![1, 2], vec![2]);

        let gather1 = tensor.gather(0, &indices1).unwrap();
        let gather2 = tensor.gather(1, &indices2).unwrap();

        assert_eq!(gather1.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(gather2.data(), &[2.0, 3.0, 5.0, 6.0]);

        // Test scatter after gather
        let scatter_values = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
        let indices = vec![&Tensor::from_vec(vec![0, 1], vec![2])];
        let scatter_result = tensor.scatter(0, &indices[0], &scatter_values).unwrap();
        assert_eq!(scatter_result.data(), &[10.0, 2.0, 3.0, 20.0, 5.0, 6.0]);
    }

    /// Test indexing operations with different tensor shapes
    #[test]
    fn test_indexing_different_shapes() {
        // Test 1D tensor indexing
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let indices_1d = Tensor::from_vec(vec![0, 2], vec![2]);
        let gather_1d = tensor_1d.gather(0, &indices_1d).unwrap();
        assert_eq!(gather_1d.data(), &[1.0, 3.0]);

        // Test 3D tensor indexing
        let tensor_3d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let indices_3d = Tensor::from_vec(vec![0, 1], vec![2]);
        let gather_3d = tensor_3d.gather(0, &indices_3d).unwrap();
        assert_eq!(gather_3d.shape(), &[2, 2, 2]);

        // Test 4D tensor indexing
        let tensor_4d = Tensor::from_vec((1..16).map(|x| x as f64).collect(), vec![2, 2, 2, 2]);
        let indices_4d = Tensor::from_vec(vec![0, 1], vec![2]);
        let gather_4d = tensor_4d.gather(0, &indices_4d).unwrap();
        assert_eq!(gather_4d.shape(), &[2, 2, 2, 2]);
    }

    /// Test index operations with negative indices
    #[test]
    fn test_negative_indices() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test negative gather indices
        let negative_indices = Tensor::from_vec(vec![-2, -1], vec![2]);
        let negative_gather = tensor.gather(0, &negative_indices);
        assert!(negative_gather.is_err()); // Should error on negative indices
    }
}

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Slice the tensor using the provided slice specifications
    ///
    /// # Arguments
    /// * `slices` - Array of slice specifications for each dimension
    ///
    /// # Returns
    /// Result containing the sliced tensor or an error if slicing is invalid
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    /// use coeus_tensor::ops::indexing::Slice;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let slice = tensor.slice(&[Slice::Range(0..1), Slice::Range(0..2)]).unwrap();
    /// // slice contains first row: [1.0, 2.0]
    /// ```
    pub fn slice(&self, slices: &[Slice]) -> Result<Tensor<T>> {
        if slices.len() != self.shape.len() {
            return Err(TensorError::InvalidOperation {
                message: format!("Number of slices ({}) must match tensor dimensions ({})",
                               slices.len(), self.shape.len())
            });
        }

        // For simplicity, implement basic 2D slicing
        if self.shape.len() == 2 && slices.len() == 2 {
            let row_slice = &slices[0];
            let col_slice = &slices[1];

            let (start_row, end_row) = match row_slice {
                Slice::Range(range) => (range.start, range.end.min(self.shape[0])),
                _ => return Err(TensorError::InvalidOperation {
                    message: "Only range slicing supported for now".to_string()
                }),
            };

            let (start_col, end_col) = match col_slice {
                Slice::Range(range) => (range.start, range.end.min(self.shape[1])),
                _ => return Err(TensorError::InvalidOperation {
                    message: "Only range slicing supported for now".to_string()
                }),
            };

            let new_rows = end_row - start_row;
            let new_cols = end_col - start_col;
            let mut result_data = Vec::with_capacity(new_rows * new_cols);

            for r in start_row..end_row {
                for c in start_col..end_col {
                    result_data.push(self.data[r * self.shape[1] + c]);
                }
            }

            Ok(Tensor::from_vec(result_data, vec![new_rows, new_cols]))
        } else {
            Err(TensorError::InvalidOperation {
                message: "Slicing only implemented for 2D tensors".to_string()
            })
        }
    }

    /// Gather elements along a dimension using indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to gather
    /// * `indices` - Tensor containing indices to gather
    ///
    /// # Returns
    /// Result containing the gathered tensor or an error if gathering fails
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let indices = Tensor::from_vec(vec![0i32, 2], vec![2]);
    /// let gathered = tensor.gather(1, &indices).unwrap();
    /// // gathered contains [1.0, 3.0] from the first row
    /// ```
    pub fn gather(&self, dim: usize, indices: &Tensor<i32>) -> Result<Tensor<T>> {
        if dim >= self.shape.len() {
            return Err(TensorError::InvalidOperation {
                message: format!("Dimension {} is out of bounds for tensor with {} dimensions",
                               dim, self.shape.len())
            });
        }

        // Calculate output shape: same as input except dim dimension becomes size of indices
        let mut output_shape = self.shape.clone();
        output_shape[dim] = indices.data.len();

        // Calculate strides for the input tensor
        let mut strides = vec![1; self.shape.len()];
        for i in (0..self.shape.len()-1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }

        let mut result_data = Vec::new();

        // Iterative implementation for 2D tensors
        if self.shape.len() == 2 && indices.shape.len() == 1 {
            if dim == 0 {
                // Gather along rows
                for &idx in &indices.data {
                    let row_idx = idx as usize;
                    if row_idx >= self.shape[0] {
                        return Err(TensorError::InvalidOperation {
                            message: format!("Index {} is out of bounds for dimension 0 with size {}",
                                           row_idx, self.shape[0])
                        });
                    }
                    for col in 0..self.shape[1] {
                        let input_idx = row_idx * self.shape[1] + col;
                        result_data.push(self.data[input_idx].clone());
                    }
                }
            } else if dim == 1 {
                // Gather along columns
                for row in 0..self.shape[0] {
                    for &idx in &indices.data {
                        let col_idx = idx as usize;
                        if col_idx >= self.shape[1] {
                            return Err(TensorError::InvalidOperation {
                                message: format!("Index {} is out of bounds for dimension 1 with size {}",
                                               col_idx, self.shape[1])
                            });
                        }
                        let input_idx = row * self.shape[1] + col_idx;
                        result_data.push(self.data[input_idx].clone());
                    }
                }
            }
        } else {
            // For now, only support 2D tensors with 1D indices
            return Err(TensorError::InvalidOperation {
                message: "Gather currently only supports 2D tensors with 1D indices".to_string()
            });
        }

        Ok(Tensor::from_vec(result_data, output_shape))
    }

    /// Scatter elements to specific positions
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Tensor containing indices where to scatter
    /// * `src` - Source tensor containing values to scatter
    ///
    /// # Returns
    /// Result containing the tensor with scattered values or an error if scattering fails
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::zeros(vec![2, 3]);
    /// let indices = Tensor::from_vec(vec![0i32, 2], vec![2]);
    /// let src = Tensor::from_vec(vec![10.0, 20.0], vec![2]);
    /// let scattered = tensor.scatter(1, &indices, &src).unwrap();
    /// // scattered has values 10.0 and 20.0 at column indices 0 and 2
    /// ```
    pub fn scatter(&self, dim: usize, indices: &Tensor<i32>, src: &Tensor<T>) -> Result<Tensor<T>> {
        if dim >= self.shape.len() {
            return Err(TensorError::InvalidOperation {
                message: format!("Dimension {} is out of bounds for tensor with {} dimensions",
                               dim, self.shape.len())
            });
        }

        // For now, only support 2D tensors with 1D indices
        if self.shape.len() != 2 || indices.shape.len() != 1 {
            return Err(TensorError::InvalidOperation {
                message: "Scatter currently only supports 2D tensors with 1D indices".to_string()
            });
        }

        // Copy the original data
        let mut result_data = self.data.clone();

        if dim == 0 {
            // Scatter along rows
            for i in 0..indices.data.len() {
                let row_idx = indices.data[i] as usize;
                if row_idx >= self.shape[0] {
                    return Err(TensorError::InvalidOperation {
                        message: format!("Index {} is out of bounds for dimension 0 with size {}",
                                       row_idx, self.shape[0])
                    });
                }

                // Copy entire row from src to result
                for col in 0..self.shape[1] {
                    let src_idx = i * self.shape[1] + col;
                    let result_idx = row_idx * self.shape[1] + col;
                    if src_idx < src.data.len() && result_idx < result_data.len() {
                        result_data[result_idx] = src.data[src_idx].clone();
                    }
                }
            }
        } else if dim == 1 {
            // Scatter along columns
            for row in 0..self.shape[0] {
                for i in 0..indices.data.len() {
                    let col_idx = indices.data[i] as usize;
                    if col_idx >= self.shape[1] {
                        return Err(TensorError::InvalidOperation {
                            message: format!("Index {} is out of bounds for dimension 1 with size {}",
                                           col_idx, self.shape[1])
                        });
                    }

                    let src_idx = row * indices.data.len() + i;
                    let result_idx = row * self.shape[1] + col_idx;
                    if src_idx < src.data.len() && result_idx < result_data.len() {
                        result_data[result_idx] = src.data[src_idx].clone();
                    }
                }
            }
        }

        Ok(Tensor::from_vec(result_data, self.shape.clone()))
    }

    /// Select elements along a dimension using indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to select
    /// * `indices` - Array of indices to select
    ///
    /// # Returns
    /// Result containing the selected tensor or an error if selection fails
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let selected = tensor.index_select(1, &[0, 2]).unwrap();
    /// // selected contains columns 0 and 2: [[1.0, 3.0], [4.0, 6.0]]
    /// ```
    pub fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T>> {
        if dim >= self.shape.len() {
            return Err(TensorError::InvalidOperation {
                message: format!("Dimension {} is out of bounds for tensor with {} dimensions",
                               dim, self.shape.len())
            });
        }

        // Check for out-of-bounds indices
        for &idx in indices {
            if idx >= self.shape[dim] {
                return Err(TensorError::InvalidOperation {
                    message: format!("Index {} is out of bounds for dimension {} with size {}",
                                   idx, dim, self.shape[dim])
                });
            }
        }

        // Calculate output shape
        let mut output_shape = self.shape.clone();
        output_shape[dim] = indices.len();

        let mut result_data = Vec::new();

        // Implementation for 2D tensors
        if self.shape.len() == 2 {
            if dim == 0 {
                // Select rows
                for &row_idx in indices {
                    for col in 0..self.shape[1] {
                        let input_idx = row_idx * self.shape[1] + col;
                        result_data.push(self.data[input_idx].clone());
                    }
                }
            } else if dim == 1 {
                // Select columns
                for row in 0..self.shape[0] {
                    for &col_idx in indices {
                        let input_idx = row * self.shape[1] + col_idx;
                        result_data.push(self.data[input_idx].clone());
                    }
                }
            }
        } else {
            return Err(TensorError::InvalidOperation {
                message: "Index select currently only supports 2D tensors".to_string()
            });
        }

        Ok(Tensor::from_vec(result_data, output_shape))
    }

    /// Perform advanced indexing with multiple index tensors
    ///
    /// # Arguments
    /// * `indices` - Array of tensors containing indices for each dimension
    ///
    /// # Returns
    /// Result containing the indexed tensor or an error if indexing fails
    pub fn advanced_index(&self, indices: &[&Tensor<i32>]) -> Result<Tensor<T>> {
        // Advanced indexing requires multi-dimensional index tensors
        // Implementation would require significant work for PyTorch compatibility
        // Defer to future sprint with full SRS specification
        Err(TensorError::InvalidOperation {
            message: "Advanced indexing requires multi-dimensional index tensors - not yet implemented".to_string()
        })
    }

    /// Put values at specific indices (tensor[index] = value)
    ///
    /// # Arguments
    /// * `indices` - Array of tensors containing indices for each dimension
    /// * `values` - Values to put at the specified indices
    ///
    /// # Returns
    /// Result containing the modified tensor or an error if operation fails
    pub fn index_put(&self, indices: &[&Tensor<i32>], values: &Tensor<T>) -> Result<Tensor<T>> {
        // Support 2D indexing
        if indices.len() != 2 || indices[0].shape.len() != 1 || indices[1].shape.len() != 1 {
            return Err(TensorError::InvalidOperation {
                message: "Index put currently only supports 2D tensors with 1D indices per dimension".to_string()
            });
        }

        if indices[0].data.len() != indices[1].data.len() {
            return Err(TensorError::InvalidOperation {
                message: "Index tensors must have the same length".to_string()
            });
        }

        if indices[0].data.len() != values.data.len() {
            return Err(TensorError::InvalidOperation {
                message: "Number of indices must match number of values".to_string()
            });
        }

        let mut result_data = self.data.clone();

        for i in 0..indices[0].data.len() {
            let row_idx = indices[0].data[i] as usize;
            let col_idx = indices[1].data[i] as usize;

            if row_idx >= self.shape[0] || col_idx >= self.shape[1] {
                return Err(TensorError::InvalidOperation {
                    message: format!("Index ({}, {}) is out of bounds for tensor with shape {:?}",
                                   row_idx, col_idx, self.shape)
                });
            }

            let flat_idx = row_idx * self.shape[1] + col_idx;
            result_data[flat_idx] = values.data[i].clone();
        }

        Ok(Tensor::from_vec(result_data, self.shape.clone()))
    }

    /// Add values at specific indices (tensor[index] += value)
    ///
    /// # Arguments
    /// * `indices` - Array of tensors containing indices for each dimension
    /// * `values` - Values to add at the specified indices
    ///
    /// # Returns
    /// Result containing the modified tensor or an error if operation fails
    pub fn index_add(&self, indices: &[&Tensor<i32>], values: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: std::ops::Add<Output = T>,
    {
        // Support 2D indexing
        if indices.len() != 2 || indices[0].shape.len() != 1 || indices[1].shape.len() != 1 {
            return Err(TensorError::InvalidOperation {
                message: "Index add currently only supports 2D tensors with 1D indices per dimension".to_string()
            });
        }

        if indices[0].data.len() != indices[1].data.len() {
            return Err(TensorError::InvalidOperation {
                message: "Index tensors must have the same length".to_string()
            });
        }

        if indices[0].data.len() != values.data.len() {
            return Err(TensorError::InvalidOperation {
                message: "Number of indices must match number of values".to_string()
            });
        }

        let mut result_data = self.data.clone();

        for i in 0..indices[0].data.len() {
            let row_idx = indices[0].data[i] as usize;
            let col_idx = indices[1].data[i] as usize;

            if row_idx >= self.shape[0] || col_idx >= self.shape[1] {
                return Err(TensorError::InvalidOperation {
                    message: format!("Index ({}, {}) is out of bounds for tensor with shape {:?}",
                                   row_idx, col_idx, self.shape)
                });
            }

            let flat_idx = row_idx * self.shape[1] + col_idx;
            result_data[flat_idx] = result_data[flat_idx].clone() + values.data[i].clone();
        }

        Ok(Tensor::from_vec(result_data, self.shape.clone()))
    }

    /// Copy values to specific indices (tensor[index] = value, but creates new tensor)
    ///
    /// # Arguments
    /// * `indices` - Array of tensors containing indices for each dimension
    /// * `values` - Values to copy to the specified indices
    ///
    /// # Returns
    /// Result containing the new tensor with copied values or an error if operation fails
    pub fn index_copy(&self, indices: &[&Tensor<i32>], values: &Tensor<T>) -> Result<Tensor<T>> {
        // For now, delegate to index_put since the semantics are similar
        self.index_put(indices, values)
    }
}
