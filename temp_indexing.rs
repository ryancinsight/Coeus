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

use crate::{Dtype, Result, Tensor, Backend, TensorError};

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
pub trait Indexing<T: Dtype, B: Backend<T> + Clone + Send + Sync> {
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
    fn slice(&self, slices: &[Slice]) -> Result<Tensor<T, B>>;

    /// Gather values along a dimension using indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to gather
    /// * `indices` - Indices to gather (slice of i64 values)
    ///
    /// # Returns
    /// Gathered tensor with same shape as indices
    fn gather(&self, dim: usize, indices: &[i64]) -> Result<Tensor<T, B>>;

    /// Scatter values to specific positions along a dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Target indices for scattering
    /// * `src` - Source tensor containing values to scatter
    ///
    /// # Returns
    /// Tensor with scattered values
    fn scatter(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>>;

    /// Scatter and add values to specific positions along a dimension
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Target indices for scattering
    /// * `src` - Source tensor containing values to add
    ///
    /// # Returns
    /// Tensor with added values at specified positions
    fn scatter_add(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>>;

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
        indices: &[i64],
        src: &Tensor<T, B>,
        reduce: &str,
    ) -> Result<Tensor<T, B>>;

    /// Select elements along a dimension by indices
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to select
    /// * `indices` - Array of indices to select
    ///
    /// # Returns
    /// Tensor with selected elements
    fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T, B>>;

    /// Advanced indexing with multiple index arrays
    ///
    /// # Arguments
    /// * `indices` - Array of index slices for each dimension
    ///
    /// # Returns
    /// Tensor with advanced indexing applied
    fn advanced_index(&self, indices: &[&[i32]]) -> Result<Tensor<T, B>>;

    /// Put values at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to put values
    /// * `values` - Values to put
    ///
    /// # Returns
    /// Tensor with values placed at specified indices
    fn index_put(&self, indices: &[&[i32]], values: &Tensor<T, B>) -> Result<Tensor<T, B>>;

    /// Add values at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to add values
    /// * `values` - Values to add
    ///
    /// # Returns
    /// Tensor with values added at specified indices
    fn index_add(&self, indices: &[&[i32]], values: &Tensor<T, B>) -> Result<Tensor<T, B>>;

    /// Copy values from source tensor at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices where to copy values
    /// * `src` - Source tensor to copy from
    ///
    /// # Returns
    /// Tensor with values copied at specified indices
    fn index_copy(&self, indices: &[&[i32]], src: &Tensor<T, B>) -> Result<Tensor<T, B>>;

    /// Fill tensor positions with a value
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to fill
    /// * `index` - Index along the dimension
    /// * `value` - Value to fill
    ///
    /// # Returns
    /// Tensor with filled values
    fn index_fill(&self, dim: usize, index: usize, value: T) -> Result<Tensor<T, B>>;

    /// Fill tensor positions based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    /// Tensor with filled values where mask is true
    fn masked_fill(&self, mask: &[bool], value: T) -> Result<Tensor<T, B>>;

    /// Scatter values based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    /// * `src` - Source tensor
    ///
    /// # Returns
    /// Tensor with values scattered where mask is true
    fn masked_scatter(&self, mask: &[bool], src: &Tensor<T, B>) -> Result<Tensor<T, B>>;

    /// Select elements based on boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean mask
    ///
    /// # Returns
    /// 1D tensor with elements where mask is true
    fn masked_select(&self, mask: &[bool]) -> Result<Tensor<T, B>>;

    /// Create a narrowed view of the tensor
    ///
    /// # Arguments
    /// * `dim` - Dimension to narrow
    /// * `start` - Starting index
    /// * `length` - Length of the narrowed view
    ///
    /// # Returns
    /// Narrowed tensor view
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T, B>>;

    /// Find indices of non-zero elements
    ///
    /// # Returns
    /// Vector of vectors containing indices of non-zero elements [dim][indices]
    fn nonzero(&self) -> Result<Vec<Vec<i64>>>;

    /// Take elements from tensor at specified indices
    ///
    /// # Arguments
    /// * `indices` - Indices to take
    ///
    /// # Returns
    /// Tensor with elements at specified indices
    fn take(&self, indices: &[i64]) -> Result<Tensor<T, B>>;

    /// Put values at specified positions (alias for index_put)
    ///
    /// # Arguments
    /// * `indices` - Indices where to put values
    /// * `values` - Values to put
    ///
    /// # Returns
    /// Tensor with values placed at specified indices
    fn put(&self, indices: &[i64], values: &Tensor<T, B>) -> Result<Tensor<T, B>>;
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

        Ok(Tensor::from_vec(self.backend().clone(), result_data, new_shape)?)
    }
}

impl<T: Dtype + Clone, B: Backend<T> + Clone + Send + Sync> Indexing<T, B> for Tensor<T, B> {
    fn slice(&self, indices: &[Slice]) -> Result<Tensor<T, B>> {
        self.backend.slice(self, indices)
    }
        for src_idx in 0..src_size {
