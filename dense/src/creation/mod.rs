//! Dense storage creation operations
//!
//! This module provides functions for creating dense storage with various
//! initialization patterns. These are basic creation operations that work
//! at the storage level.

pub mod zeros;
pub mod ones;
pub mod from_vec;

// Re-export creation functions
pub use zeros::{zeros, zeros_like, scalar_zero};
pub use ones::{ones, ones_like, scalar_one, eye, eye_rectangular};
pub use from_vec::{from_vec, from_slice, full, full_like, from_iter, from_fn, scalar};

use storage::{DenseStorage, Result};
use dtype::DataType;
use alloc::vec::Vec;

/// Trait for dense storage creation operations
///
/// This trait provides a unified interface for creating dense storage
/// with various initialization patterns.
pub trait DenseCreation<T: DataType> {
    /// Create storage filled with zeros
    fn zeros(shape: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::Zero;

    /// Create storage filled with ones
    fn ones(shape: &[usize]) -> Result<Self>
    where
        Self: Sized,
        T: num_traits::One;

    /// Create storage filled with constant value
    fn full(shape: &[usize], value: T) -> Result<Self>
    where
        Self: Sized,
        T: Clone;

    /// Create storage from vector
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self>
    where
        Self: Sized;
}

impl<T: DataType> DenseCreation<T> for DenseStorage<T> {
    fn zeros(shape: &[usize]) -> Result<Self>
    where
        T: num_traits::Zero,
    {
        zeros::zeros(shape)
    }

    fn ones(shape: &[usize]) -> Result<Self>
    where
        T: num_traits::One,
    {
        ones::ones(shape)
    }

    fn full(shape: &[usize], value: T) -> Result<Self>
    where
        T: Clone,
    {
        from_vec::full(shape, value)
    }

    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        from_vec::from_vec(data, shape)
    }
}