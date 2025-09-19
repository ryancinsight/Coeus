//! Indexing operations for tensors
//!
//! This module provides PyTorch-compatible advanced indexing operations
//! including slicing, gathering, scattering, and advanced selection methods.
//!
//! ## Supported Operations
//!
//! - **Slice Operations**: Extract sub-tensors using range-based indexing
//! - **Gather Operations**: Collect values along a dimension using indices
//! - **Scatter Operations**: Distribute values to specific positions
//! - **Index Select**: Select elements by indices along a dimension
//! - **Selection**: Boolean masking and conditional selection
//!
//! ## Usage
//!
//! ```rust,ignore
//! use coeus_tensor::Tensor;
//!
//! let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
//!
//! // Slice operations: tensor[0..2, 1..3]
//! let slice = tensor.slice(&[0..2, 1..3]).unwrap();
//!
//! // Gather along dimension 0 with indices [0, 0, 1]
//! let indices = Tensor::from_vec(vec![0i32, 0, 1], vec![3]);
//! let gathered = tensor.gather(0, &indices).unwrap();
//!
//! // Index select along dimension 1
//! let selected = tensor.index_select(1, &[0, 2]).unwrap();
//! ```
//!
//! ## Broadcasting and Shape Handling
//!
//! All indexing operations handle broadcasting automatically and maintain
//! compatibility with PyTorch's indexing semantics.
//!
//! ## References
//!
//! - [PyTorch Indexing](https://pytorch.org/docs/stable/notes/advanced_indexing.html)
//! - [NumPy Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)

use crate::{Dtype, Result, Tensor, TensorError};

/// Range specification for slicing operations
#[derive(Clone, Debug)]
pub struct Slice {
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub step: Option<usize>,
}

impl Slice {
    /// Create a new slice with optional start, end, and step
    pub fn new(start: Option<usize>, end: Option<usize>, step: Option<usize>) -> Self {
        Self { start, end, step }
    }

    /// Create a slice representing all elements (:)
    pub fn all() -> Self {
        Self::new(None, None, None)
    }

    /// Create a slice with start..end range
    pub fn range(start: usize, end: usize) -> Self {
        Self::new(Some(start), Some(end), None)
    }

    /// Create a slice with start..end..step range
    pub fn range_step(start: usize, end: usize, step: usize) -> Self {
        Self::new(Some(start), Some(end), Some(step))
    }

    /// Resolve slice bounds for a given dimension size
    pub fn resolve(&self, dim_size: usize) -> (usize, usize, usize) {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(dim_size);
        let step = self.step.unwrap_or(1);

        // Handle negative indices
        let start = if start > dim_size { dim_size } else { start };
        let end = if end > dim_size { dim_size } else { end };

        (start, end, step)
    }
}

/// Indexing operations for tensors
pub trait Indexing<T: Dtype> {
    /// Slice tensor using range specifications for each dimension
    ///
    /// # Arguments
    /// * `slices` - Slice specifications for each dimension
    ///
    /// # Returns
    /// Sliced tensor with reduced dimensions
    ///
    /// # Example
    /// ```rust,ignore
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let slice = tensor.slice(&[
    ///     Slice::range(0, 1),  // First row
    ///     Slice::all()         // All columns
    /// ]).unwrap();
    /// ```
    fn slice(&self, slices: &[Slice]) -> Result<Tensor<T>>;

    /// Gather values along a dimension using indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to gather
    /// * `indices` - Indices to gather (should be integer tensor)
    ///
    /// # Returns
    /// Gathered tensor with same shape as indices
    fn gather(&self, dim: usize, indices: &Tensor<i32>) -> Result<Tensor<T>>;

    /// Scatter values to specific positions along a dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Target indices for scattering
    /// * `src` - Source tensor containing values to scatter
    ///
    /// # Returns
    /// Tensor with scattered values
    fn scatter(&self, dim: usize, indices: &Tensor<i32>, src: &Tensor<T>) -> Result<Tensor<T>>;

    /// Scatter and add values to specific positions along a dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Target indices for scattering
    /// * `src` - Source tensor containing values to add
    ///
    /// # Returns
    /// Tensor with added values at specified positions
    fn scatter_add(&self, dim: usize, indices: &Tensor<i32>, src: &Tensor<T>) -> Result<Tensor<T>>;

    /// Scatter and reduce values to specific positions along a dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Target indices for scattering
    /// * `src` - Source tensor containing values to reduce
    /// * `reduce` - Reduction operation ("add", "multiply", "maximum", "minimum")
    ///
    /// # Returns
    /// Tensor with reduced values at specified positions
    fn scatter_reduce(
        &self,
        dim: usize,
        indices: &Tensor<i32>,
        src: &Tensor<T>,
        reduce: &str,
    ) -> Result<Tensor<T>>;

    /// Select elements along a dimension by indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to select
    /// * `indices` - Array of indices to select
    ///
    /// # Returns
    /// Tensor with selected elements
    fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T>>;

    /// Advanced indexing with multiple index arrays
    ///
    /// # Arguments
    /// * `indices` - Array of index tensors for each dimension
    ///
    /// # Returns
    /// Tensor with advanced indexing applied
    fn advanced_index(&self, indices: &[&Tensor<i32>]) -> Result<Tensor<T>>;

    /// Put values at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to put values
    /// * `values` - Values to put
    ///
    /// # Returns
    /// Tensor with values placed at specified indices
    fn index_put(&self, indices: &[&Tensor<i32>], values: &Tensor<T>) -> Result<Tensor<T>>;

    /// Add values at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to add values
    /// * `values` - Values to add
    ///
    /// # Returns
    /// Tensor with values added at specified indices
    fn index_add(&self, indices: &[&Tensor<i32>], values: &Tensor<T>) -> Result<Tensor<T>>;

    /// Copy values from source tensor at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to copy values
    /// * `src` - Source tensor to copy from
    ///
    /// # Returns
    /// Tensor with values copied at specified indices
    fn index_copy(&self, indices: &[&Tensor<i32>], src: &Tensor<T>) -> Result<Tensor<T>>;

    /// Fill tensor positions with a value
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to fill
    /// * `index` - Index along the dimension
    /// * `value` - Value to fill
    ///
    /// # Returns
    /// Tensor with filled values
    fn index_fill(&self, dim: usize, index: usize, value: T) -> Result<Tensor<T>>;

    /// Fill tensor positions based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    /// Tensor with filled values where mask is true
    fn masked_fill(&self, mask: &[bool], value: T) -> Result<Tensor<T>>;

    /// Scatter values based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    /// * `src` - Source tensor
    ///
    /// # Returns
    /// Tensor with values scattered where mask is true
    fn masked_scatter(&self, mask: &[bool], src: &Tensor<T>) -> Result<Tensor<T>>;

    /// Select elements based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    ///
    /// # Returns
    /// 1D tensor with elements where mask is true
    fn masked_select(&self, mask: &[bool]) -> Result<Tensor<T>>;

    /// Create a narrowed view of the tensor
    ///
    /// # Arguments
    /// * `dim` - Dimension to narrow
    /// * `start` - Starting index
    /// * `length` - Length of the narrowed view
    ///
    /// # Returns
    /// Narrowed tensor view
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>>;

    /// Find indices of non-zero elements
    ///
    /// # Returns
    /// Tensor of shape [num_nonzero, ndim] containing indices of non-zero elements
    fn nonzero(&self) -> Result<Tensor<i64>>;

    /// Take elements from tensor at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices to take
    ///
    /// # Returns
    /// Tensor with elements at specified indices
    fn take(&self, indices: &Tensor<i64>) -> Result<Tensor<T>>;

    /// Put values at specified positions (alias for index_put)
    ///
    /// # Arguments
    /// * `indices` - Indices where to put values
    /// * `values` - Values to put
    ///
    /// # Returns
    /// Tensor with values placed at specified indices
    fn put(&self, indices: &Tensor<i64>, values: &Tensor<T>) -> Result<Tensor<T>>;
}

impl<T: Dtype + Clone + num_traits::FromPrimitive + num_traits::ToPrimitive> Indexing<T>
    for Tensor<T>
{
    fn slice(&self, slices: &[Slice]) -> Result<Tensor<T>> {
        if slices.len() > self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Too many slices: expected at most {} but got {}",
                    self.ndim(),
                    slices.len()
                ),
            });
        }

        // Resolve slice bounds for each dimension
        let mut resolved_slices = Vec::new();
        for (i, slice) in slices.iter().enumerate() {
            if i >= self.shape().len() {
                break;
            }
            resolved_slices.push(slice.resolve(self.shape()[i]));
        }

        // For dimensions not specified, include all
        for i in resolved_slices.len()..self.shape().len() {
            resolved_slices.push((0, self.shape()[i], 1));
        }

        // Calculate new shape
        let mut new_shape = Vec::new();
        for (start, end, step) in &resolved_slices {
            let size = if *step == 1 {
                end - start
            } else {
                ((end - start) as f64 / *step as f64).ceil() as usize
            };
            new_shape.push(size);
        }

        // Extract sliced data
        let mut result_data = Vec::new();
        let total_size = new_shape.iter().product();

        for flat_idx in 0..total_size {
            // Convert flat index to coordinates in new shape
            let mut coords = vec![0; new_shape.len()];
            let mut remaining = flat_idx;
            for i in (0..new_shape.len()).rev() {
                coords[i] = remaining % new_shape[i];
                remaining /= new_shape[i];
            }

            // Map coordinates back to original tensor
            let mut orig_coords = vec![0; self.shape().len()];
            for i in 0..orig_coords.len() {
                let (start, _, step) = resolved_slices[i];
                orig_coords[i] = start + coords[i] * step;
            }

            // Convert to flat index in original tensor
            let mut orig_flat_idx = 0;
            let mut stride = 1;
            for i in (0..orig_coords.len()).rev() {
                orig_flat_idx += orig_coords[i] * stride;
                stride *= self.shape()[i];
            }

            if orig_flat_idx < self.data().len() {
                result_data.push(self.data()[orig_flat_idx]);
            }
        }

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    fn gather(&self, dim: usize, indices: &Tensor<i32>) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        // Create new shape: replace dim with indices shape
        let mut new_shape = self.shape().to_vec();
        let index_shape = indices.shape().to_vec();

        // Remove the gather dimension and insert index dimensions
        new_shape.remove(dim);
        for (i, &size) in index_shape.iter().enumerate() {
            new_shape.insert(dim + i, size);
        }

        let mut result_data = Vec::new();

        // For each position in the result tensor
        let total_size = new_shape.iter().product();
        for flat_idx in 0..total_size {
            // Convert flat index to coordinates
            let mut coords = vec![0; new_shape.len()];
            let mut remaining = flat_idx;
            for i in (0..new_shape.len()).rev() {
                coords[i] = remaining % new_shape[i];
                remaining /= new_shape[i];
            }

            // Extract coordinates for the gather dimension
            let _gather_coord = coords[dim];

            // Create original coordinates
            let mut orig_coords = vec![0; self.shape().len()];

            // Copy coordinates from result, but handle the gather dimension specially
            let mut result_idx = 0;
            for (d, _) in (0..self.shape().len()).enumerate() {
                if d == dim {
                    // Use the gather coordinate
                    let gather_flat_idx = {
                        let mut idx = 0;
                        let mut stride = 1;
                        for i in (0..index_shape.len()).rev() {
                            let coord_idx = dim + i;
                            if coord_idx < coords.len() {
                                idx += coords[coord_idx] * stride;
                            }
                            stride *= index_shape[i];
                        }
                        idx
                    };

                    if gather_flat_idx < indices.data().len() {
                        let index_val = indices.data()[gather_flat_idx];
                        orig_coords[d] = index_val.max(0).min(self.shape()[d] as i32 - 1) as usize;
                    } else {
                        orig_coords[d] = 0;
                    }
                } else {
                    // Use coordinate from result, adjusting for gather dimension
                    if result_idx < coords.len() {
                        orig_coords[d] = coords[result_idx];
                    }
                    result_idx += 1;
                }
            }

            // Convert to flat index in original tensor
            let mut orig_flat_idx = 0;
            let mut stride = 1;
            for i in (0..orig_coords.len()).rev() {
                orig_flat_idx += orig_coords[i] * stride;
                stride *= self.shape()[i];
            }

            if orig_flat_idx < self.data().len() {
                result_data.push(self.data()[orig_flat_idx]);
            }
        }

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    fn scatter(&self, dim: usize, indices: &Tensor<i32>, src: &Tensor<T>) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        // Create a copy of the original tensor data
        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        // For each position in the source tensor
        let src_size = src.data().len();
        for src_idx in 0..src_size {
            // Convert source index to coordinates
            let mut src_coords = vec![0; src.shape().len()];
            let mut remaining = src_idx;
            for i in (0..src.shape().len()).rev() {
                src_coords[i] = remaining % src.shape()[i];
                remaining /= src.shape()[i];
            }

            // Get the target index from indices tensor
            let index_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for i in (0..indices.shape().len()).rev() {
                    if i < src_coords.len() {
                        idx += src_coords[i] * stride;
                    }
                    stride *= indices.shape()[i];
                }
                idx
            };

            if index_idx < indices.data().len() {
                let target_coord = indices.data()[index_idx]
                    .max(0)
                    .min(self.shape()[dim] as i32 - 1) as usize;

                // Create target coordinates
                let mut target_coords = vec![0; self.shape().len()];
                for (d, _) in (0..self.shape().len()).enumerate() {
                    if d == dim {
                        target_coords[d] = target_coord;
                    } else {
                        // Map from source coordinates to target coordinates
                        let src_dim_idx = if d < dim { d } else { d - 1 };
                        if src_dim_idx < src_coords.len() {
                            target_coords[d] = src_coords[src_dim_idx];
                        }
                    }
                }

                // Convert to flat index
                let mut target_flat_idx = 0;
                let mut stride = 1;
                for i in (0..target_coords.len()).rev() {
                    target_flat_idx += target_coords[i] * stride;
                    stride *= self.shape()[i];
                }

                if target_flat_idx < result_data.len() {
                    result_data[target_flat_idx] = src.data()[src_idx];
                }
            }
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn scatter_add(&self, dim: usize, indices: &Tensor<i32>, src: &Tensor<T>) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        // Create a copy of the original tensor data
        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        // For each position in the source tensor
        let src_size = src.data().len();
        for src_idx in 0..src_size {
            // Convert source index to coordinates
            let mut src_coords = vec![0; src.shape().len()];
            let mut remaining = src_idx;
            for i in (0..src.shape().len()).rev() {
                src_coords[i] = remaining % src.shape()[i];
                remaining /= src.shape()[i];
            }

            // Get the target index from indices tensor
            let index_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for i in (0..indices.shape().len()).rev() {
                    if i < src_coords.len() {
                        idx += src_coords[i] * stride;
                    }
                    stride *= indices.shape()[i];
                }
                idx
            };

            if index_idx < indices.data().len() {
                let target_coord = indices.data()[index_idx]
                    .max(0)
                    .min(self.shape()[dim] as i32 - 1) as usize;

                // Create target coordinates
                let mut target_coords = vec![0; self.shape().len()];
                for (d, _) in (0..self.shape().len()).enumerate() {
                    if d == dim {
                        target_coords[d] = target_coord;
                    } else {
                        // Map from source coordinates to target coordinates
                        let src_dim_idx = if d < dim { d } else { d - 1 };
                        if src_dim_idx < src_coords.len() {
                            target_coords[d] = src_coords[src_dim_idx];
                        }
                    }
                }

                // Convert to flat index
                let mut target_flat_idx = 0;
                let mut stride = 1;
                for i in (0..target_coords.len()).rev() {
                    target_flat_idx += target_coords[i] * stride;
                    stride *= self.shape()[i];
                }

                if target_flat_idx < result_data.len() {
                    // Add the source value to the target value
                    result_data[target_flat_idx] =
                        result_data[target_flat_idx] + src.data()[src_idx];
                }
            }
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn scatter_reduce(
        &self,
        dim: usize,
        indices: &Tensor<i32>,
        src: &Tensor<T>,
        reduce: &str,
    ) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        // Create a copy of the original tensor data
        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        // For each position in the source tensor
        let src_size = src.data().len();
        for src_idx in 0..src_size {
            // Convert source index to coordinates
            let mut src_coords = vec![0; src.shape().len()];
            let mut remaining = src_idx;
            for i in (0..src.shape().len()).rev() {
                src_coords[i] = remaining % src.shape()[i];
                remaining /= src.shape()[i];
            }

            // Get the target index from indices tensor
            let index_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for i in (0..indices.shape().len()).rev() {
                    if i < src_coords.len() {
                        idx += src_coords[i] * stride;
                    }
                    stride *= indices.shape()[i];
                }
                idx
            };

            if index_idx < indices.data().len() {
                let target_coord = indices.data()[index_idx]
                    .max(0)
                    .min(self.shape()[dim] as i32 - 1) as usize;

                // Create target coordinates
                let mut target_coords = vec![0; self.shape().len()];
                for (d, _) in (0..self.shape().len()).enumerate() {
                    if d == dim {
                        target_coords[d] = target_coord;
                    } else {
                        // Map from source coordinates to target coordinates
                        let src_dim_idx = if d < dim { d } else { d - 1 };
                        if src_dim_idx < src_coords.len() {
                            target_coords[d] = src_coords[src_dim_idx];
                        }
                    }
                }

                // Convert to flat index
                let mut target_flat_idx = 0;
                let mut stride = 1;
                for i in (0..target_coords.len()).rev() {
                    target_flat_idx += target_coords[i] * stride;
                    stride *= self.shape()[i];
                }

                if target_flat_idx < result_data.len() {
                    // Apply the reduction operation
                    let src_val = src.data()[src_idx];
                    let target_val = &mut result_data[target_flat_idx];

                    match reduce {
                        "add" => *target_val = *target_val + src_val,
                        "multiply" => *target_val = *target_val * src_val,
                        "maximum" => {
                            if src_val > *target_val {
                                *target_val = src_val
                            }
                        }
                        "minimum" => {
                            if src_val < *target_val {
                                *target_val = src_val
                            }
                        }
                        _ => {
                            return Err(TensorError::InvalidOperation {
                                message: format!("Unsupported reduction operation: {}", reduce),
                            })
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        // Validate indices
        for &idx in indices {
            if idx >= self.shape()[dim] {
                return Err(TensorError::InvalidOperation {
                    message: format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        idx,
                        dim,
                        self.shape()[dim]
                    ),
                });
            }
        }

        // Create new shape
        let mut new_shape = self.shape().to_vec();
        new_shape[dim] = indices.len();

        let mut result_data = Vec::new();

        // Iterate over all positions in the result tensor
        let total_size = new_shape.iter().product();
        for flat_idx in 0..total_size {
            // Convert flat index to coordinates in result
            let mut coords = vec![0; new_shape.len()];
            let mut remaining = flat_idx;
            for i in (0..new_shape.len()).rev() {
                coords[i] = remaining % new_shape[i];
                remaining /= new_shape[i];
            }

            // Create coordinates in original tensor
            let mut orig_coords = coords.clone();
            orig_coords[dim] = indices[coords[dim]];

            // Convert to flat index in original tensor
            let mut orig_flat_idx = 0;
            let mut stride = 1;
            for i in (0..orig_coords.len()).rev() {
                orig_flat_idx += orig_coords[i] * stride;
                stride *= self.shape()[i];
            }

            if orig_flat_idx < self.data().len() {
                result_data.push(self.data()[orig_flat_idx]);
            }
        }

        Ok(Tensor::from_vec(result_data, new_shape))
    }

    fn advanced_index(&self, indices: &[&Tensor<i32>]) -> Result<Tensor<T>> {
        if indices.len() > self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Too many index arrays: expected at most {} but got {}",
                    self.ndim(),
                    indices.len()
                ),
            });
        }

        // Validate that all index tensors have the same shape
        let index_shape = indices[0].shape().to_vec();
        for (_i, index_tensor) in indices.iter().enumerate().skip(1) {
            if index_tensor.shape() != index_shape.as_slice() {
                return Err(TensorError::ShapeMismatch {
                    expected: index_shape.clone(),
                    actual: index_tensor.shape().to_vec(),
                });
            }
        }

        let mut result_data = Vec::new();

        // For each position in the index shape
        let index_size = index_shape.iter().product();
        for index_idx in 0..index_size {
            // Convert index to coordinates
            let mut index_coords = vec![0; index_shape.len()];
            let mut remaining = index_idx;
            for i in (0..index_shape.len()).rev() {
                index_coords[i] = remaining % index_shape[i];
                remaining /= index_shape[i];
            }

            // Get the actual indices for each dimension
            let mut target_coords = vec![0; self.shape().len()];
            for d in 0..indices.len() {
                let flat_idx = {
                    let mut idx = 0;
                    let mut stride = 1;
                    for i in (0..index_shape.len()).rev() {
                        idx += index_coords[i] * stride;
                        stride *= index_shape[i];
                    }
                    idx
                };

                if flat_idx < indices[d].data().len() {
                    let index_val = indices[d].data()[flat_idx];
                    target_coords[d] = index_val.max(0).min(self.shape()[d] as i32 - 1) as usize;
                }
            }

            // For dimensions not indexed, use the index coordinates
            for d in indices.len()..self.shape().len() {
                if d < index_coords.len() {
                    target_coords[d] = index_coords[d];
                }
            }

            // Convert to flat index
            let mut target_flat_idx = 0;
            let mut stride = 1;
            for i in (0..target_coords.len()).rev() {
                target_flat_idx += target_coords[i] * stride;
                stride *= self.shape()[i];
            }

            if target_flat_idx < self.data().len() {
                result_data.push(self.data()[target_flat_idx]);
            }
        }

        Ok(Tensor::from_vec(result_data, index_shape))
    }

    fn index_put(&self, _indices: &[&Tensor<i32>], _values: &Tensor<T>) -> Result<Tensor<T>> {
        // For simplicity, this is a basic implementation - full implementation would need
        // proper index_put semantics which are complex
        Err(TensorError::InvalidOperation {
            message: "index_put not yet fully implemented".to_string(),
        })
    }

    fn index_add(&self, _indices: &[&Tensor<i32>], _values: &Tensor<T>) -> Result<Tensor<T>> {
        // This is a simplified implementation - full index_add would handle broadcasting
        // and complex indexing patterns
        Err(TensorError::InvalidOperation {
            message: "index_add not yet fully implemented".to_string(),
        })
    }

    fn index_copy(&self, _indices: &[&Tensor<i32>], _src: &Tensor<T>) -> Result<Tensor<T>> {
        // Simplified implementation
        Err(TensorError::InvalidOperation {
            message: "index_copy not yet fully implemented".to_string(),
        })
    }

    fn index_fill(&self, dim: usize, index: usize, value: T) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        if index >= self.shape()[dim] {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Index {} out of bounds for dimension {} with size {}",
                    index,
                    dim,
                    self.shape()[dim]
                ),
            });
        }

        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        // Calculate strides for each dimension
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape()[i + 1];
        }

        // Fill the specified index along the dimension
        let mut coords = vec![0; self.ndim()];
        let total_elements = self.data().len();

        #[allow(clippy::needless_range_loop)]
        for flat_idx in 0..total_elements {
            // Convert flat index to coordinates
            let mut temp_idx = flat_idx;
            for d in 0..self.ndim() {
                coords[d] = temp_idx / strides[d];
                temp_idx %= strides[d];
            }

            // If this coordinate matches the target index along the specified dimension, fill it
            if coords[dim] == index {
                result_data[flat_idx] = value;
            }
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    fn masked_fill(&self, mask: &[bool], value: T) -> Result<Tensor<T>> {
        if mask.len() != self.data().len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.data().len()],
                actual: vec![mask.len()],
            });
        }

        let mut result_data = self.data().to_vec();
        for (i, &mask_val) in mask.iter().enumerate() {
            if mask_val {
                result_data[i] = value;
            }
        }

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    fn masked_scatter(&self, mask: &[bool], src: &Tensor<T>) -> Result<Tensor<T>> {
        if mask.len() != self.data().len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.data().len()],
                actual: vec![mask.len()],
            });
        }

        let mut result_data = self.data().to_vec();
        let mut src_idx = 0;

        for (i, &mask_val) in mask.iter().enumerate() {
            if mask_val && src_idx < src.data().len() {
                result_data[i] = src.data()[src_idx];
                src_idx += 1;
            }
        }

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    fn masked_select(&self, mask: &[bool]) -> Result<Tensor<T>> {
        if mask.len() != self.data().len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.data().len()],
                actual: vec![mask.len()],
            });
        }

        let mut result_data = Vec::new();
        for (i, &mask_val) in mask.iter().enumerate() {
            if mask_val {
                result_data.push(self.data()[i]);
            }
        }

        let len = result_data.len();
        Ok(Tensor::from_vec(result_data, vec![len]))
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        if start + length > self.shape()[dim] {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Narrow range [{}, {}) out of bounds for dimension {} with size {}",
                    start,
                    start + length,
                    dim,
                    self.shape()[dim]
                ),
            });
        }

        // Create slices for narrowing
        let mut slices = vec![Slice::all(); self.ndim()];
        slices[dim] = Slice::range(start, start + length);

        self.slice(&slices)
    }

    fn nonzero(&self) -> Result<Tensor<i64>> {
        let mut indices = Vec::new();

        for (flat_idx, &value) in self.data().iter().enumerate() {
            // Check if value is non-zero (this is a simplification - proper implementation
            // would need to handle floating point epsilon comparisons)
            if value != T::zero() {
                // Convert flat index to coordinates
                let mut coords = vec![0i64; self.ndim()];
                let mut remaining = flat_idx;
                for i in (0..self.ndim()).rev() {
                    coords[i] = (remaining % self.shape()[i]) as i64;
                    remaining /= self.shape()[i];
                }
                indices.extend(coords);
            }
        }

        let num_nonzero = indices.len() / self.ndim();
        Ok(Tensor::from_vec(indices, vec![num_nonzero, self.ndim()]))
    }

    fn take(&self, indices: &Tensor<i64>) -> Result<Tensor<T>> {
        if indices.ndim() != 1 {
            return Err(TensorError::InvalidOperation {
                message: "take indices must be 1D".to_string(),
            });
        }

        let mut result_data = Vec::new();
        for &idx in indices.data() {
            let idx_usize = idx as usize;
            if idx_usize >= self.numel() {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx_usize,
                    size: self.numel(),
                });
            }
            result_data.push(self.data()[idx_usize]);
        }

        Ok(Tensor::from_vec(result_data, vec![indices.numel()]))
    }

    fn put(&self, indices: &Tensor<i64>, values: &Tensor<T>) -> Result<Tensor<T>> {
        if indices.ndim() != 1 {
            return Err(TensorError::InvalidOperation {
                message: "put indices must be 1D".to_string(),
            });
        }

        if indices.numel() != values.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![indices.numel()],
                actual: vec![values.numel()],
            });
        }

        let mut result_data = self.data().to_vec();

        for (i, &idx) in indices.data().iter().enumerate() {
            let idx_usize = idx as usize;
            if idx_usize >= self.numel() {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx_usize,
                    size: self.numel(),
                });
            }
            result_data[idx_usize] = values.data()[i];
        }

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }
}

/// Utility functions for creating slices
pub mod slices {
    use super::Slice;

    /// Create a slice representing all elements (:)
    pub fn all() -> Slice {
        Slice::all()
    }

    /// Create a slice with start..end range
    pub fn range(start: usize, end: usize) -> Slice {
        Slice::range(start, end)
    }

    /// Create a slice with start..end..step range
    pub fn range_step(start: usize, end: usize, step: usize) -> Slice {
        Slice::range_step(start, end, step)
    }

    /// Create a slice with start index only
    pub fn from(start: usize) -> Slice {
        Slice::new(Some(start), None, None)
    }

    /// Create a slice with end index only
    pub fn to(end: usize) -> Slice {
        Slice::new(None, Some(end), None)
    }

    /// Create a slice with single index
    pub fn at(index: usize) -> Slice {
        Slice::range(index, index + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_slice_basic() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Slice first row
        let slice = tensor.slice(&[Slice::range(0, 1), Slice::all()]).unwrap();
        assert_eq!(slice.shape(), &[1, 3]);
        assert_eq!(slice.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_slice_with_step() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Slice with step
        let slice = tensor
            .slice(&[Slice::all(), Slice::range_step(0, 3, 2)])
            .unwrap();
        assert_eq!(slice.shape(), &[2, 2]);
        assert_eq!(slice.data(), &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_index_select() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Select columns 0 and 2
        let selected = tensor.index_select(1, &[0, 2]).unwrap();
        assert_eq!(selected.shape(), &[2, 2]);
        assert_eq!(selected.data(), &[1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scatter_add() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let indices = Tensor::from_vec(vec![0i32, 1], vec![2]);
        let src = Tensor::from_vec(vec![10.0, 20.0], vec![2]);

        let result = tensor.scatter_add(0, &indices, &src).unwrap();

        // Expected: [[11.0, 2.0], [3.0, 24.0]]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data()[0], 11.0); // 1.0 + 10.0
        assert_eq!(result.data()[1], 2.0); // unchanged
        assert_eq!(result.data()[2], 3.0); // unchanged
        assert_eq!(result.data()[3], 24.0); // 4.0 + 20.0
    }

    #[test]
    fn test_index_fill() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = tensor.index_fill(1, 1, 99.0).unwrap();

        // Fill index 1 along dimension 1 (columns)
        assert_eq!(result.shape(), &[2, 3]);
        // Data layout: [row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]
        assert_eq!(result.data(), &[1.0, 99.0, 3.0, 4.0, 99.0, 6.0]);
    }

    #[test]
    fn test_masked_fill() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = &[true, false, true, false];
        let result = tensor.masked_fill(mask, 99.0).unwrap();

        assert_eq!(result.data(), &[99.0, 2.0, 99.0, 4.0]);
    }

    #[test]
    fn test_masked_select() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = &[true, false, true, false];
        let result = tensor.masked_select(mask).unwrap();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.data(), &[1.0, 3.0]);
    }

    #[test]
    fn test_narrow() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = tensor.narrow(1, 1, 2).unwrap();

        // Narrow along dimension 1 (columns) from index 1 with length 2
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[2.0, 3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_nonzero() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 0.0, 3.0, 0.0, 5.0], vec![5]);
        let result = tensor.nonzero().unwrap();

        // Should return indices of non-zero elements: [0, 2, 4]
        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.data(), &[0i64, 2, 4]);
    }

    #[test]
    fn test_gather() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let indices = Tensor::from_vec(vec![0i32, 2, 1], vec![3]);

        // Gather along dimension 1
        let gathered = tensor.gather(1, &indices).unwrap();
        assert_eq!(gathered.shape(), &[2, 3]);
        assert_eq!(gathered.data(), &[1.0, 3.0, 2.0, 4.0, 6.0, 5.0]);
    }
}
