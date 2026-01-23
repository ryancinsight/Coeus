//! Basic arithmetic operations for dense storage
//!
//! This module provides element-wise arithmetic operations that form the foundation
//! of dense tensor computations. These operations delegate to backend primitives for
//! hardware-optimized execution.

pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

// Re-export all arithmetic functions
pub use add::{add, add_scalar};
pub use sub::{sub, sub_scalar, scalar_sub};
pub use mul::{mul, mul_scalar};
pub use div::{div, div_scalar, scalar_div};

use storage::{DenseStorage, Result};
use dtype::DataType;
use backend::Backend;

/// Trait for basic arithmetic operations on dense storage
///
/// This trait provides a unified interface for element-wise arithmetic operations.
/// All operations delegate to the individual function implementations for consistency.
pub trait DenseArithmetic<T: DataType> {
    /// Element-wise addition
    fn add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Add<Output = T> + Clone;

    /// Element-wise subtraction
    fn sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Sub<Output = T> + Clone;

    /// Element-wise multiplication
    fn mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Mul<Output = T> + Clone;

    /// Element-wise division
    fn div(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Div<Output = T> + Clone;

    /// Scalar addition
    fn add_scalar(&self, scalar: T) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Add<Output = T> + Clone;

    /// Scalar multiplication
    fn mul_scalar(&self, scalar: T) -> Result<Self>
    where
        Self: Sized,
        T: core::ops::Mul<Output = T> + Clone;
}

impl<T: DataType> DenseArithmetic<T> for DenseStorage<T> {
    fn add<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + Clone,
    {
        add::add(self, other, backend)
    }

    fn sub<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: core::ops::Sub<Output = T> + Clone,
    {
        sub::sub(self, other, backend)
    }

    fn mul<B: Backend<Data = T>>(&self, other: &Self, backend: &B) -> Result<Self>
    where
        T: core::ops::Mul<Output = T> + Clone,
    {
        mul::mul(self, other, backend)
    }

    fn div(&self, other: &Self) -> Result<Self>
    where
        T: core::ops::Div<Output = T> + Clone,
    {
        div::div(self, other)
    }

    fn add_scalar(&self, scalar: T) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + Clone,
    {
        add::add_scalar(self, scalar)
    }

    fn mul_scalar(&self, scalar: T) -> Result<Self>
    where
        T: core::ops::Mul<Output = T> + Clone,
    {
        mul::mul_scalar(self, scalar)
    }
}