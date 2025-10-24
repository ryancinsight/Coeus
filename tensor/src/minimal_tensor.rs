//! Minimal tensor implementation for testing.
//!
//! This provides a basic tensor implementation that compiles and allows
//! the neural network crate to be tested properly.

use std::ops::{Add, Div, Mul, Sub};
use std::vec::Vec;

/// Minimal tensor implementation for testing
#[derive(Debug, Clone)]
pub struct MinimalTensor<B, S, T> {
    data: Vec<T>,
    shape: Vec<usize>,
    _backend: B,
    _storage: S,
}

impl<B, S, T> MinimalTensor<B, S, T> {
    /// Create a new tensor
    pub fn new(data: Vec<T>, shape: Vec<usize>, backend: B, storage: S) -> Self {
        Self {
            data,
            shape,
            _backend: backend,
            _storage: storage,
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Create from vec
    pub fn from_vec(backend: B, data: Vec<T>, shape: Vec<usize>) -> Result<Self, crate::TensorError>
    where
        S: Default,
    {
        Ok(Self::new(data, shape, backend, S::default()))
    }

    /// SIMD addition
    pub fn add_simd(&self, other: &Self) -> Result<Self, crate::TensorError>
    where
        T: Copy + std::ops::Add<Output = T>,
        B: Clone,
        S: Clone,
    {
        if self.shape != other.shape {
            return Err(crate::TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
                operation: "add",
            });
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Self::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        ))
    }

    /// SIMD ReLU
    pub fn relu_simd(&self) -> Result<Self, crate::TensorError>
    where
        T: Copy + PartialOrd + num_traits::Zero,
        B: Clone,
        S: Clone,
    {
        let zero = T::zero();
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&val| if val > zero { val } else { zero })
            .collect();

        Ok(Self::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        ))
    }

    /// SIMD sum
    pub fn sum_simd(&self) -> Result<T, crate::TensorError>
    where
        T: Copy + num_traits::Zero + std::ops::Add<Output = T>,
    {
        Ok(self.data.iter().fold(T::zero(), |acc, &x| acc + x))
    }

    /// Check if tensor requires gradients (stub implementation - always false for testing)
    pub fn requires_grad(&self) -> bool {
        false
    }

    /// Get gradient (stub implementation - always None for testing)
    pub fn grad(&self) -> Result<Option<&Self>, crate::TensorError> {
        Ok(None)
    }

    /// Zero gradients (stub implementation - no-op for testing)
    pub fn zero_grad(&mut self) -> Result<(), crate::TensorError> {
        Ok(())
    }

    /// Get mutable slice access (limited implementation for testing)
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Create zeros tensor (for optimizer compatibility)
    pub fn zeros(shape: &[usize]) -> Result<Self, crate::TensorError>
    where
        T: num_traits::Zero + Clone,
        B: Default,
        S: Default,
    {
        let total_elements: usize = shape.iter().product();
        let data = vec![T::zero(); total_elements];
        Ok(Self::new(data, shape.to_vec(), B::default(), S::default()))
    }

    /// Create tensor from scalar (for optimizer compatibility)
    pub fn from_scalar(scalar: T, backend: B, storage: S) -> Self {
        Self::new(vec![scalar], vec![1], backend, storage)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Self, crate::TensorError>
    where
        T: Copy + num_traits::Float,
        B: Clone,
        S: Clone,
    {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();
        Ok(Self::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        ))
    }
}

// Implement arithmetic operations for optimizer compatibility
impl<B, S, T> Add<&MinimalTensor<B, S, T>> for &MinimalTensor<B, S, T>
where
    T: Copy + Add<Output = T>,
    B: Clone,
    S: Clone,
{
    type Output = MinimalTensor<B, S, T>;

    fn add(self, other: &MinimalTensor<B, S, T>) -> Self::Output {
        // For simplicity, assume same shape and broadcast scalar operations
        let result_data: Vec<T> = if self.data.len() == 1 && other.data.len() > 1 {
            // Broadcast self (scalar) to other's shape
            other.data.iter().map(|&x| self.data[0] + x).collect()
        } else if other.data.len() == 1 && self.data.len() > 1 {
            // Broadcast other (scalar) to self's shape
            self.data.iter().map(|&x| x + other.data[0]).collect()
        } else if self.data.len() == other.data.len() {
            // Element-wise addition
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            // Fallback - just add first elements (for testing)
            vec![self.data[0] + other.data[0]]
        };

        MinimalTensor::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        )
    }
}

impl<B, S, T> Sub<&MinimalTensor<B, S, T>> for &MinimalTensor<B, S, T>
where
    T: Copy + Sub<Output = T>,
    B: Clone,
    S: Clone,
{
    type Output = MinimalTensor<B, S, T>;

    fn sub(self, other: &MinimalTensor<B, S, T>) -> Self::Output {
        let result_data: Vec<T> = if self.data.len() == 1 && other.data.len() > 1 {
            other.data.iter().map(|&x| self.data[0] - x).collect()
        } else if other.data.len() == 1 && self.data.len() > 1 {
            self.data.iter().map(|&x| x - other.data[0]).collect()
        } else if self.data.len() == other.data.len() {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a - b)
                .collect()
        } else {
            vec![self.data[0] - other.data[0]]
        };

        MinimalTensor::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        )
    }
}

impl<B, S, T> Mul<&MinimalTensor<B, S, T>> for &MinimalTensor<B, S, T>
where
    T: Copy + Mul<Output = T>,
    B: Clone,
    S: Clone,
{
    type Output = MinimalTensor<B, S, T>;

    fn mul(self, other: &MinimalTensor<B, S, T>) -> Self::Output {
        let result_data: Vec<T> = if self.data.len() == 1 && other.data.len() > 1 {
            other.data.iter().map(|&x| self.data[0] * x).collect()
        } else if other.data.len() == 1 && self.data.len() > 1 {
            self.data.iter().map(|&x| x * other.data[0]).collect()
        } else if self.data.len() == other.data.len() {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a * b)
                .collect()
        } else {
            vec![self.data[0] * other.data[0]]
        };

        MinimalTensor::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        )
    }
}

impl<B, S, T> Div<&MinimalTensor<B, S, T>> for &MinimalTensor<B, S, T>
where
    T: Copy + Div<Output = T>,
    B: Clone,
    S: Clone,
{
    type Output = MinimalTensor<B, S, T>;

    fn div(self, other: &MinimalTensor<B, S, T>) -> Self::Output {
        let result_data: Vec<T> = if self.data.len() == 1 && other.data.len() > 1 {
            other.data.iter().map(|&x| self.data[0] / x).collect()
        } else if other.data.len() == 1 && self.data.len() > 1 {
            self.data.iter().map(|&x| x / other.data[0]).collect()
        } else if self.data.len() == other.data.len() {
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a / b)
                .collect()
        } else {
            vec![self.data[0] / other.data[0]]
        };

        MinimalTensor::new(
            result_data,
            self.shape.clone(),
            self._backend.clone(),
            self._storage.clone(),
        )
    }
}
