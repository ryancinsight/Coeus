//! Layout operations for dense storage
//!
//! This module provides functions for manipulating the memory layout and shape
//! of dense storage. These operations form the foundation of tensor reshaping
//! and view operations.

pub mod reshape;
pub mod transpose;
pub mod stride;

// Re-export layout functions
pub use reshape::{reshape, reshape_like, flatten, unflatten};
pub use transpose::{transpose, transpose_2d, swap_axes};
pub use stride::{as_strided, compute_strides, is_contiguous};

use storage::{DenseStorage, Result};
use dtype::DataType;

/// Trait for dense storage layout operations
///
/// This trait provides a unified interface for manipulating the memory layout
/// and shape of dense storage.
pub trait DenseLayout<T: DataType> {
    /// Reshape storage to new dimensions
    fn reshape(&self, new_shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Transpose storage (2D only)
    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;

    /// Flatten storage to 1D
    fn flatten(&self) -> Result<Self>
    where
        Self: Sized;

    /// Check if storage is contiguous in memory
    fn is_contiguous(&self) -> bool;
}

impl<T: DataType> DenseLayout<T> for DenseStorage<T> {
    fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        reshape::reshape(self, new_shape)
    }

    fn transpose(&self) -> Result<Self> {
        transpose::transpose_2d(self)
    }

    fn flatten(&self) -> Result<Self> {
        reshape::flatten(self)
    }

    fn is_contiguous(&self) -> bool {
        stride::is_contiguous(self)
    }
}