//! Unified storage operation traits
//!
//! This module defines a comprehensive trait hierarchy for storage operations,
//! providing a single source of truth for all storage-related functionality.

use crate::error::StorageError;
use crate::DataType;
use alloc::vec::Vec;

/// Result type for storage operations
pub type Result<T> = core::result::Result<T, StorageError>;

/// Core storage operations that all storage types must implement
pub trait StorageOps<T: DataType>: Sized {
    /// Get the number of elements in the storage
    fn len(&self) -> usize;

    /// Check if the storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the underlying data (if contiguous)
    fn as_slice(&self) -> Option<&[T]>;

    /// Get a mutable reference to the underlying data (if contiguous)
    fn as_mut_slice(&mut self) -> Option<&mut [T]>;

    /// Clone the storage
    fn clone_storage(&self) -> Self;
}



/// Transpose and reshape operations
pub trait LayoutOps<T: DataType>: StorageOps<T> {
    /// Transpose a 2D matrix
    fn transpose(&self, rows: usize, cols: usize) -> Result<Self>;

    /// Reshape the storage (must preserve total elements)
    fn reshape(&self, old_shape: &[usize], new_shape: &[usize]) -> Result<Self>;

    /// Permute dimensions
    fn permute(&self, shape: &[usize], axes: &[usize]) -> Result<Self>;
}

/// Arithmetic operations on storage
pub trait ArithmeticOps<T: DataType>: StorageOps<T> {
    /// Element-wise addition
    fn add(&self, other: &Self) -> Result<Self>;

    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> Result<Self>;

    /// Element-wise multiplication
    fn mul(&self, other: &Self) -> Result<Self>;

    /// Element-wise division
    fn div(&self, other: &Self) -> Result<Self>;

    /// Scalar addition
    fn add_scalar(&self, scalar: T) -> Result<Self>;

    /// Scalar multiplication
    fn mul_scalar(&self, scalar: T) -> Result<Self>;
}

/// Reduction operations
pub trait ReductionOps<T: DataType>: StorageOps<T> {
    /// Product of all elements
    fn product(&self) -> T;

    /// Maximum element
    fn max(&self) -> Option<T>;

    /// Minimum element
    fn min(&self) -> Option<T>;

    /// Mean of all elements
    fn mean(&self) -> T
    where
        T: num_traits::Float;
}

/// Sparse-specific operations
pub trait SparseOps<T: DataType>: StorageOps<T> {
    /// Convert to dense storage
    fn to_dense(&self, shape: &[usize]) -> Result<crate::DenseStorage<T>>;

    /// Get number of non-zero elements
    fn nnz(&self) -> usize;

    /// Get sparsity ratio (nnz / total_elements)
    fn sparsity(&self, total_elements: usize) -> f64 {
        self.nnz() as f64 / total_elements as f64
    }

    /// Check if the storage is sparse
    fn is_sparse(&self) -> bool {
        true
    }
}

/// Quantized storage operations
pub trait QuantizedOps<T: DataType>: StorageOps<T> {
    /// Dequantize to full precision
    fn dequantize(&self) -> Result<crate::DenseStorage<T>>;

    /// Get quantization scale
    fn scale(&self) -> f64;

    /// Get quantization zero point
    fn zero_point(&self) -> i32;

    /// Get number of bits per element
    fn bits_per_element(&self) -> usize;
}

/// Distributed storage operations
pub trait DistributedOps<T: DataType>: StorageOps<T> {
    /// Get the local shard
    fn local_shard(&self) -> &Self;

    /// Get the rank of this shard
    fn rank(&self) -> usize;

    /// Get the world size
    fn world_size(&self) -> usize;

    /// Gather all shards to a single storage
    fn gather(&self) -> Result<crate::DenseStorage<T>>;

    /// Scatter storage across ranks
    fn scatter(storage: &crate::DenseStorage<T>, world_size: usize) -> Result<Vec<Self>>;
}

/// Marker trait for storage types that support all operations
pub trait FullStorage<T: DataType>:
    StorageOps<T> + LayoutOps<T> + ArithmeticOps<T> + ReductionOps<T>
{
}

// Blanket implementation for types that implement all required traits
impl<T, S> FullStorage<T> for S
where
    T: DataType,
    S: StorageOps<T> + LayoutOps<T> + ArithmeticOps<T> + ReductionOps<T>,
{
}
