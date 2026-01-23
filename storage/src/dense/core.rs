//! Dense core struct and accessors

use crate::{DataType, Shape};
use alloc::vec;
use alloc::vec::Vec;

/// Dense contiguous storage with row-major layout.
///
/// Memory is allocated as a single contiguous block with elements ordered
/// in row-major (C-contiguous) format for cache-efficient access.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStorage<T: DataType> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Shape,
    pub(crate) strides: Vec<usize>,
}

impl<T: DataType> Default for DenseStorage<T> {
    fn default() -> Self {
        let shape = Shape::new(&[]).expect("Scalar shape is valid");
        Self {
            data: vec![T::default()],
            shape,
            strides: vec![],
        }
    }
}

impl<T: DataType> DenseStorage<T> {
    /// Get reference to shape
    pub fn shape_ref(&self) -> &Shape {
        &self.shape
    }

    /// Get length of storage data
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
