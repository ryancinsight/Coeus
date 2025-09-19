//! Indexing and advanced indexing operations for tensors
//!
//! This module contains operations for accessing and modifying tensor elements
//! using various indexing schemes including slicing, gathering, and scattering.

use crate::{Tensor, TensorError, Dtype, Result};
use crate::ops::indexing::Slice;

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

        // Simple iterative implementation for 2D tensors first
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

        // Simple implementation for 2D tensors
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
        // Placeholder implementation
        Err(TensorError::InvalidOperation {
            message: "Indexing operation not yet implemented".to_string()
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
        // For now, only support simple 2D indexing
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
        // For now, only support simple 2D indexing
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
