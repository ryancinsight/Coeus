//! Attention utilities and core traits.
//!
//! This module provides the fundamental traits and utilities for attention mechanisms,
//! including compile-time dispatch, marker traits, and basic attention implementations.

use std::marker::PhantomData;

use coeus_backend::Backend;
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage};
use coeus_tensor::Tensor;

use crate::error::Result;

/// Trait for compile-time dispatch of attention computation based on storage type.
///
/// This trait enables zero-cost abstraction by specializing attention computation
/// at compile-time rather than using runtime type checks. Different storage types
/// (DenseStorage, CsrStorage, etc.) get optimized implementations.
///
/// # Type Parameters
/// - `B`: Backend type
/// - `S`: Storage type (determines specialization)
/// - `T`: Data type
///
/// # Associated Types
/// - `AttentionImpl`: The concrete implementation type for this storage
///
/// # Examples
/// ```ignore
/// // Dense storage gets dense attention implementation
/// impl<B, T> AttentionDispatch<B, DenseStorage<T>, T> for MultiHeadAttention<B, DenseStorage<T>, T>
/// where
///     B: Backend<Data = T> + Clone + Default,
///     T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
/// {
///     type AttentionImpl = DenseAttention<B, T>;
/// }
/// ```
pub trait AttentionDispatch<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// The specialized attention implementation for this storage type
    type AttentionImpl;

    /// Get the specialized implementation for this storage type
    fn get_specialized_impl(&self) -> &Self::AttentionImpl;

    /// Compute attention using the specialized implementation
    fn compute_specialized(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
}

/// Marker trait for dense storage types that should use dense attention
pub trait DenseStorageMarker {}

/// Marker trait for sparse storage types that should use sparse attention
pub trait SparseStorageMarker {}

impl<T: DataType> DenseStorageMarker for DenseStorage<T> {}

// Note: Sparse storage markers would be implemented in the storage crate
// For now, we use runtime detection as a fallback

/// Dense attention implementation optimized for contiguous memory layouts
#[derive(Debug, Clone)]
pub struct DenseAttention<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    _phantom: PhantomData<(B, T)>,
}

impl<B, T> Default for DenseAttention<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, T> DenseAttention<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Create a new dense attention implementation
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

/// Sparse attention implementation optimized for sparse matrix operations
#[derive(Debug, Clone)]
pub struct SparseAttentionImpl<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    _phantom: PhantomData<(B, T)>,
}

impl<B, T> Default for SparseAttentionImpl<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, T> SparseAttentionImpl<B, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Create a new sparse attention implementation
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}
