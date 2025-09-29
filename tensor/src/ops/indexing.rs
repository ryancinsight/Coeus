//! Indexing operations for tensors following PyTorch API
//!
//! This module provides comprehensive indexing operations including
//! select, gather, scatter, and advanced indexing capabilities.

use crate::{Tensor, TensorError, Dtype, Backend, BackendData, Result as TensorResult};
use std::sync::Arc;

// Define ReduceOp for scatter_reduce operations
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
}

/// Represents a slice operation for tensor indexing
#[derive(Debug, Clone)]
pub enum Slice {
    Full,
    Range(usize, usize),
}

/// Select operation: Select values at specified indices
///
/// # Arguments
/// * `tensor` - Source tensor
/// * `indices` - Indices to select from the flattened tensor
///
/// # Returns
/// New tensor with selected values
///
/// # Example
/// ```rust
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
/// let result = select(&tensor, &[0, 2])?;  // [1.0, 3.0]
/// ```
pub fn select<T: Dtype + Clone, B: Backend<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B>,
    indices: &[usize],
) -> TensorResult<Tensor<T, B>> {
    let data: Vec<T> = indices
        .iter()
        .map(|&i| {
            if i >= tensor.numel() {
                return Err(TensorError::IndexOutOfBounds {
                    index: i,
                    size: tensor.numel(),
                });
            }
            Ok(tensor.data()[i].clone())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let shape = vec![indices.len()];
    let backend = tensor.backend().clone();
    Ok(Tensor::from_vec(backend, data, shape)?)
}

/// Trait for advanced indexing operations on tensors
pub trait Indexing<T: Dtype, B: Backend<T> + Clone + Send + Sync> {
    /// Create a slice with the given indices
    fn slice(&self, indices: &[Slice]) -> Result<Tensor<T, B>, TensorError>;

    /// Select elements along a dimension using indices
    fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T, B>, TensorError>;

    /// Gather values from specific positions along a dimension
    fn gather(&self, dim: usize, indices: &[i64]) -> Result<Tensor<T, B>, TensorError>;

    /// Scatter values to specific positions along a dimension
    fn scatter(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError>;

    /// Scatter and add values to specific positions along a dimension
    fn scatter_add(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError>;

    /// Scatter and reduce values to specific positions along a dimension
    fn scatter_reduce(
        &self,
        dim: usize,
        indices: &[i64],
        src: &Tensor<T, B>,
        reduce: ReduceOp,
    ) -> Result<Tensor<T, B>, TensorError>;

    /// Scatter values to positions specified by a mask
    fn masked_scatter(&self, mask: &[bool], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError>;
}

impl<T: Dtype + Clone, B: Backend<T> + Clone + Send + Sync> Tensor<T, B> {
    #[allow(dead_code)]
    fn slice_impl(&self, indices: &[Slice]) -> Result<Tensor<T, B>, TensorError> {
        // Basic single-dim slice for now
        if indices.len() != 1 {
            return Err(TensorError::InvalidOperation { message: "Multi-dim slice stubbed".to_string() });
        }
        let slice = &indices[0];
        let dim = 0; // Stub single dim
        let shape = self.shape();
        let size_dim = shape[dim];
        let (start, len) = match slice {
            Slice::Full => (0, size_dim),
            Slice::Range(s, l) => (*s, *l),
        };
        if start + len > size_dim {
            return Err(TensorError::IndexOutOfBounds { index: start + len, size: size_dim });
        }
        let offset = start;
        let data_slice = &self.data()[offset..offset + len];
        let backend_data = self.backend().create_tensor_data(data_slice.to_vec(), vec![len])?;
        Ok(Tensor::from_backend_data(self.backend().clone(), Arc::new(backend_data)))
    }

    fn index_select_impl(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T, B>, TensorError> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidDimension { dim, max_dim: self.ndim() });
        }
        let shape = self.shape();
        let size_dim = shape[dim];
        let other_size = shape.iter().enumerate().filter(|&(i,_)| i != dim).map(|(_, &s)| s).product::<usize>();
        let mut result_data = vec![T::zero(); indices.len() * other_size];
        let data = self.data();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= size_dim {
                return Err(TensorError::IndexOutOfBounds { index: idx, size: size_dim });
            }
            let offset = i * size_dim + idx;
            // Copy slice for other dims (simplified, assume 1D for now)
            for j in 0..other_size {
                result_data[i * other_size + j] = data[offset + j * size_dim];
            }
        }
        let mut result_shape = shape.to_vec();
        result_shape[dim] = indices.len();
        let backend_data = self.backend().create_tensor_data(result_data, result_shape.clone())?;
        Ok(Tensor::from_backend_data(self.backend().clone(), Arc::new(backend_data)))
    }

    fn gather_impl(&self, dim: usize, indices: &[i64]) -> Result<Tensor<T, B>, TensorError> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidDimension { dim, max_dim: self.ndim() });
        }
        let shape = self.shape();
        let size_dim = shape[dim];
        let indices_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        if indices_usize.iter().any(|&i| i >= size_dim) {
            return Err(TensorError::IndexOutOfBounds { index: *indices_usize.iter().max().unwrap_or(&0), size: size_dim });
        }
        let other_size = shape.iter().enumerate().filter(|&(i,_)| i != dim).map(|(_, &s)| s).product::<usize>();
        let mut result_data = vec![T::zero(); indices.len() * other_size];
        let data = self.data();
        for (i, &idx) in indices_usize.iter().enumerate() {
            let offset = i * size_dim + idx;
            for j in 0..other_size {
                result_data[i * other_size + j] = data[offset + j * size_dim];
            }
        }
        let mut result_shape = shape.to_vec();
        result_shape[dim] = indices.len();
        let backend_data = self.backend().create_tensor_data(result_data, result_shape.clone())?;
        Ok(Tensor::from_backend_data(self.backend().clone(), Arc::new(backend_data)))
    }
}

impl<T: Dtype + Clone + std::ops::AddAssign, B: Backend<T> + Clone + Send + Sync> Indexing<T, B> for Tensor<T, B> {
    #[tracing::instrument]
    fn slice(&self, indices: &[Slice]) -> Result<Self, TensorError> {
        // Assume single dim for basic impl, multi-dim future
        let (start, end) = match indices.first() {
            Some(Slice::Range(s, e)) => (*s as usize, *e as usize),
            _ => return Err(TensorError::UnsupportedIndex(indices.iter().cloned().collect())),
        };
        if start >= end || end > self.numel() {
            return Err(TensorError::InvalidIndex { start, end, numel: self.numel() });
        }
        // View via backend (CPU manual Cow)
        let view_data = Arc::new(BackendData::Cpu {
            data: self.data().to_vec()[start..end].to_vec(), // Temp, future Cow::Borrowed(&self.data()[start..end])
            shape: vec![end - start],
        });
        Ok(Self::from_backend_data(self.backend().clone(), view_data))
    }

    fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor<T, B>, TensorError> {
        self.index_select_impl(dim, indices)
    }

    fn gather(&self, dim: usize, indices: &[i64]) -> Result<Tensor<T, B>, TensorError> {
        self.gather_impl(dim, indices)
    }

    fn scatter(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        let mut result_data = vec![T::zero(); self.data().len()];
        let result_shape = self.shape().to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            let idx_usize = if idx < 0 {
                (idx + self.shape()[dim] as i64) as usize
            } else {
                idx as usize
            };

            if idx_usize >= self.shape()[dim] {
                return Err(TensorError::OutOfBounds {
                    index: idx as usize,
                    size: self.shape()[dim],
                });
            }

            let mut src_coords = vec![0; self.ndim()];
            let mut remaining = i;
            for d in (0..self.ndim()).rev() {
                if d == dim {
                    src_coords[d] = idx_usize;
                } else {
                    src_coords[d] = remaining % self.shape()[d];
                    remaining /= self.shape()[d];
                }
            }

            let src_flat_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for d in (0..self.ndim()).rev() {
                    idx += src_coords[d] * stride;
                    stride *= self.shape()[d];
                }
                idx
            };

            if src_flat_idx < src.data().len() {
                result_data[src_flat_idx] = src.data()[src_flat_idx];
            }
        }

        Ok(Tensor::from_vec(self.backend().clone(), result_data, result_shape)?)
    }

    fn scatter_add(&self, dim: usize, indices: &[i64], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            let idx_usize = if idx < 0 {
                (idx + self.shape()[dim] as i64) as usize
            } else {
                idx as usize
            };

            if idx_usize >= self.shape()[dim] {
                return Err(TensorError::OutOfBounds {
                    index: idx as usize,
                    size: self.shape()[dim],
                });
            }

            let mut src_coords = vec![0; self.ndim()];
            let mut remaining = i;
            for d in (0..self.ndim()).rev() {
                if d == dim {
                    src_coords[d] = idx_usize;
                } else {
                    src_coords[d] = remaining % self.shape()[d];
                    remaining /= self.shape()[d];
                }
            }

            let src_flat_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for d in (0..self.ndim()).rev() {
                    idx += src_coords[d] * stride;
                    stride *= self.shape()[d];
                }
                idx
            };

            if src_flat_idx < src.data().len() {
                result_data[src_flat_idx] += src.data()[src_flat_idx];
            }
        }

        Ok(Tensor::from_vec(self.backend().clone(), result_data, result_shape)?)
    }

    fn scatter_reduce(
        &self,
        dim: usize,
        indices: &[i64],
        src: &Tensor<T, B>,
        reduce: ReduceOp,
    ) -> Result<Tensor<T, B>, TensorError> {
        if dim >= self.ndim() {
            return Err(TensorError::InvalidOperation {
                message: format!(
                    "Dimension {} out of bounds for {}D tensor",
                    dim,
                    self.ndim()
                ),
            });
        }

        let mut result_data = self.data().to_vec();
        let result_shape = self.shape().to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            let idx_usize = if idx < 0 {
                (idx + self.shape()[dim] as i64) as usize
            } else {
                idx as usize
            };

            if idx_usize >= self.shape()[dim] {
                return Err(TensorError::OutOfBounds {
                    index: idx as usize,
                    size: self.shape()[dim],
                });
            }

            let mut src_coords = vec![0; self.ndim()];
            let mut remaining = i;
            for d in (0..self.ndim()).rev() {
                if d == dim {
                    src_coords[d] = idx_usize;
                } else {
                    src_coords[d] = remaining % self.shape()[d];
                    remaining /= self.shape()[d];
                }
            }

            let src_flat_idx = {
                let mut idx = 0;
                let mut stride = 1;
                for d in (0..self.ndim()).rev() {
                    idx += src_coords[d] * stride;
                    stride *= self.shape()[d];
                }
                idx
            };

            if src_flat_idx < src.data().len() {
                let target_value = &mut result_data[src_flat_idx];
                let src_value = src.data()[src_flat_idx];
                *target_value = match reduce {
                    ReduceOp::Add => *target_value + src_value,
                    ReduceOp::Mul => *target_value * src_value,
                    ReduceOp::Min => if *target_value < src_value { *target_value } else { src_value },
                    ReduceOp::Max => if *target_value > src_value { *target_value } else { src_value },
                };
            }
        }

        Ok(Tensor::from_vec(self.backend().clone(), result_data, result_shape)?)
    }

    fn masked_scatter(&self, mask: &[bool], src: &Tensor<T, B>) -> Result<Tensor<T, B>, TensorError> {
        if mask.len() != self.data().len() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: vec![mask.len()],
            });
        }

        let mut result_data = self.data().to_vec();
        let mut src_idx = 0;

        for (i, &masked) in mask.iter().enumerate() {
            if masked {
                if src_idx < src.data().len() {
                    result_data[i] = src.data()[src_idx];
                    src_idx += 1;
                }
            }
        }

        Ok(Tensor::from_vec(self.backend().clone(), result_data, self.shape().to_vec())?)
    }
}

