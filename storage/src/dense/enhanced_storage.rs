//! Enhanced dense storage implementation with zero-cost abstractions
//!
//! This module provides the enhanced dense storage implementation that supports
//! the new trait hierarchy for compile-time dispatch.

use crate::enhanced_traits::*;
use crate::{DataType, Shape, StorageError};
use alloc::vec::Vec;
use core::fmt;

/// Enhanced dense storage implementation
///
/// Provides contiguous row-major memory layout with zero-cost access patterns.
/// Optimized for cache-friendly operations and SIMD vectorization.
#[derive(Clone, Debug, PartialEq)]
pub struct EnhancedDenseStorage<T: DataType> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: DataType> EnhancedDenseStorage<T> {
    /// Create new dense storage with given shape, filled with default values
    pub fn new(shape: &[usize]) -> crate::Result<Self> {
        let total_elements = shape.iter().product();
        Ok(Self {
            data: vec![T::default(); total_elements],
            shape: Shape::new(shape)?,
        })
    }
    
    /// Create new dense storage filled with zeros
    pub fn zeros(shape: &[usize]) -> crate::Result<Self> {
        let total_elements = shape.iter().product();
        Ok(Self {
            data: vec![T::zero(); total_elements],
            shape: Shape::new(shape)?,
        })
    }
    
    /// Create new dense storage filled with ones
    pub fn ones(shape: &[usize]) -> crate::Result<Self>
    where
        T: num_traits::One,
    {
        let total_elements = shape.iter().product();
        Ok(Self {
            data: vec![T::one(); total_elements],
            shape: Shape::new(shape)?,
        })
    }
    
    /// Create new dense storage filled with a specific value
    pub fn full(shape: &[usize], value: T) -> crate::Result<Self> {
        let total_elements = shape.iter().product();
        Ok(Self {
            data: vec![value; total_elements],
            shape: Shape::new(shape)?,
        })
    }
    
    /// Get element at linear index (unchecked for performance)
    ///
    /// # Safety
    /// Caller must ensure index is within bounds
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.data.get_unchecked(index)
    }
    
    /// Get mutable element at linear index (unchecked for performance)
    ///
    /// # Safety
    /// Caller must ensure index is within bounds
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.data.get_unchecked_mut(index)
    }
    
    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let linear_index = self.shape.linear_index(indices).ok()?;
        self.data.get(linear_index)
    }
    
    /// Get mutable element at multi-dimensional index
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let linear_index = self.shape.linear_index(indices).ok()?;
        self.data.get_mut(linear_index)
    }
    
    /// Reshape the storage (must preserve total elements)
    pub fn reshape(&self, new_shape: &[usize]) -> crate::Result<Self> {
        let new_total = new_shape.iter().product::<usize>();
        if new_total != self.data.len() {
            return Err(StorageError::InvalidShape(format!(
                "Cannot reshape from {} to {} elements",
                self.data.len(),
                new_total
            )));
        }
        
        Ok(Self {
            data: self.data.clone(),
            shape: Shape::new(new_shape)?,
        })
    }
    
    /// Create a view of a slice of the data
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.data.get(start..end)
    }
    
    /// Create a mutable view of a slice of the data
    pub fn slice_mut(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        self.data.get_mut(start..end)
    }
}

// Core Storage trait implementation
impl<T: DataType> Storage<T> for EnhancedDenseStorage<T> {
    fn shape(&self) -> &Shape {
        &self.shape
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn nnz(&self) -> usize {
        // For dense storage, count actual non-zero elements
        self.data.iter().filter(|&&x| x != T::zero()).count()
    }
}

// Dense storage trait implementation
impl<T: DataType> DenseStorage<T> for EnhancedDenseStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    fn from_vec(data: Vec<T>, shape: &[usize]) -> crate::Result<Self> {
        let expected_len = shape.iter().product::<usize>();
        if data.len() != expected_len {
            return Err(StorageError::InvalidShape(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }
        
        Ok(Self {
            data,
            shape: Shape::new(shape)?,
        })
    }
    
    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

// Storage creation trait implementation
impl<T: DataType> StorageFromVec<T> for EnhancedDenseStorage<T> {
    fn from_vec(data: Vec<T>, shape: &[usize]) -> crate::Result<Self> {
        <Self as DenseStorage<T>>::from_vec(data, shape)
    }
}

// Storage info trait implementation
impl<T: DataType> StorageInfo<T> for EnhancedDenseStorage<T> {
    fn storage_type(&self) -> StorageType {
        StorageType::Dense
    }
    
    fn memory_usage(&self) -> usize {
        self.data.len() * core::mem::size_of::<T>() + core::mem::size_of::<Shape>()
    }
    
    fn is_optimal_for(&self, _operation: &str) -> bool {
        true // Dense is optimal for all operations
    }
}

// Arithmetic operations implementation
impl<T: DataType> StorageArithmetic<T> for EnhancedDenseStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Div<Output = T>
        + Copy,
{
    fn add(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot add tensors with shapes {:?} and {:?}",
                self.shape.dims(),
                other.shape.dims()
            )));
        }
        
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn sub(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot subtract tensors with shapes {:?} and {:?}",
                self.shape.dims(),
                other.shape.dims()
            )));
        }
        
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn mul(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot multiply tensors with shapes {:?} and {:?}",
                self.shape.dims(),
                other.shape.dims()
            )));
        }
        
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn div(&self, other: &Self) -> crate::Result<Self> {
        if self.shape != other.shape {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot divide tensors with shapes {:?} and {:?}",
                self.shape.dims(),
                other.shape.dims()
            )));
        }
        
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn add_scalar(&self, scalar: T) -> crate::Result<Self> {
        let result_data: Vec<T> = self.data.iter().map(|&x| x + scalar).collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn mul_scalar(&self, scalar: T) -> crate::Result<Self> {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
}

// Linear algebra operations implementation
impl<T: DataType> StorageLinearAlgebra<T> for EnhancedDenseStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + num_traits::Zero
        + Copy,
{
    fn matmul(&self, other: &Self) -> crate::Result<Self> {
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();
        
        if self_dims.len() != 2 || other_dims.len() != 2 {
            return Err(StorageError::InvalidOperation(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }
        
        let (m, k) = (self_dims[0], self_dims[1]);
        let (k2, n) = (other_dims[0], other_dims[1]);
        
        if k != k2 {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot multiply matrices with shapes [{}, {}] and [{}, {}]",
                m, k, k2, n
            )));
        }
        
        let mut result_data = vec![T::zero(); m * n];
        
        // Optimized matrix multiplication with cache-friendly access pattern
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + self.data[i * k + l] * other.data[l * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }
        
        Self::from_vec(result_data, &[m, n])
    }
    
    fn matvec(&self, vec: &[T]) -> crate::Result<Vec<T>> {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            return Err(StorageError::InvalidOperation(
                "Matrix-vector multiplication requires 2D matrix".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        if vec.len() != n {
            return Err(StorageError::ShapeMismatch(format!(
                "Cannot multiply matrix with shape [{}, {}] by vector of length {}",
                m, n, vec.len()
            )));
        }
        
        let mut result = vec![T::zero(); m];
        
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + self.data[i * n + j] * vec[j];
            }
            result[i] = sum;
        }
        
        Ok(result)
    }
    
    fn transpose(&self) -> crate::Result<Self> {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            return Err(StorageError::InvalidOperation(
                "Transpose requires 2D tensor".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        let mut result_data = vec![T::zero(); m * n];
        
        for i in 0..m {
            for j in 0..n {
                result_data[j * m + i] = self.data[i * n + j];
            }
        }
        
        Self::from_vec(result_data, &[n, m])
    }
}

// Reduction operations implementation
impl<T: DataType> StorageReduction<T> for EnhancedDenseStorage<T>
where
    T: core::ops::Add<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::Zero
        + num_traits::FromPrimitive
        + Copy,
{
    fn sum(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x)
    }
    
    fn mean(&self) -> T
    where
        T: num_traits::Float,
    {
        if self.data.is_empty() {
            return T::zero();
        }
        
        let sum = self.sum();
        let len = T::from(self.data.len()).unwrap_or(T::one());
        sum / len
    }
    
    fn max(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        self.data.iter().max().copied()
    }
    
    fn min(&self) -> Option<T>
    where
        T: PartialOrd,
    {
        self.data.iter().min().copied()
    }
    
    fn sum_axis(&self, axis: usize) -> crate::Result<Self> {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(StorageError::InvalidOperation(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis, dims.len()
            )));
        }
        
        // For now, implement simple case for 2D tensors
        if dims.len() == 2 {
            match axis {
                0 => {
                    // Sum along rows (result is 1D with shape [cols])
                    let (rows, cols) = (dims[0], dims[1]);
                    let mut result = vec![T::zero(); cols];
                    
                    for j in 0..cols {
                        for i in 0..rows {
                            result[j] = result[j] + self.data[i * cols + j];
                        }
                    }
                    
                    Self::from_vec(result, &[cols])
                }
                1 => {
                    // Sum along columns (result is 1D with shape [rows])
                    let (rows, cols) = (dims[0], dims[1]);
                    let mut result = vec![T::zero(); rows];
                    
                    for i in 0..rows {
                        for j in 0..cols {
                            result[i] = result[i] + self.data[i * cols + j];
                        }
                    }
                    
                    Self::from_vec(result, &[rows])
                }
                _ => unreachable!(),
            }
        } else {
            Err(StorageError::InvalidOperation(
                "sum_axis not yet implemented for non-2D tensors".to_string(),
            ))
        }
    }
    
    fn mean_axis(&self, axis: usize) -> crate::Result<Self>
    where
        T: num_traits::Float,
    {
        let sum_result = self.sum_axis(axis)?;
        let dims = self.shape.dims();
        let axis_size = T::from(dims[axis]).unwrap_or(T::one());
        
        sum_result.mul_scalar(T::one() / axis_size)
    }
}

// Activation operations implementation
impl<T: DataType> StorageActivation<T> for EnhancedDenseStorage<T>
where
    T: PartialOrd + num_traits::Zero + num_traits::Float + Copy,
{
    fn relu(&self) -> crate::Result<Self>
    where
        T: PartialOrd + num_traits::Zero,
    {
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn sigmoid(&self) -> crate::Result<Self>
    where
        T: num_traits::Float,
    {
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| T::one() / (T::one() + (-x).exp()))
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn tanh(&self) -> crate::Result<Self>
    where
        T: num_traits::Float,
    {
        let result_data: Vec<T> = self.data.iter().map(|&x| x.tanh()).collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn gelu(&self) -> crate::Result<Self>
    where
        T: num_traits::Float,
    {
        let sqrt_2_pi = T::from(2.0 / core::f64::consts::PI).unwrap().sqrt();
        let coeff = T::from(0.044715).unwrap();
        
        let result_data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                let x_cubed = x * x * x;
                let inner = sqrt_2_pi * (x + coeff * x_cubed);
                x * T::from(0.5).unwrap() * (T::one() + inner.tanh())
            })
            .collect();
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    fn softmax(&self, axis: usize) -> crate::Result<Self>
    where
        T: num_traits::Float,
    {
        let dims = self.shape.dims();
        if axis >= dims.len() {
            return Err(StorageError::InvalidOperation(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis, dims.len()
            )));
        }
        
        // For now, implement simple case for 2D tensors
        if dims.len() == 2 && axis == 1 {
            let (rows, cols) = (dims[0], dims[1]);
            let mut result_data = vec![T::zero(); rows * cols];
            
            for i in 0..rows {
                // Find max for numerical stability
                let mut max_val = self.data[i * cols];
                for j in 1..cols {
                    let val = self.data[i * cols + j];
                    if val > max_val {
                        max_val = val;
                    }
                }
                
                // Compute exp(x - max) and sum
                let mut sum = T::zero();
                for j in 0..cols {
                    let exp_val = (self.data[i * cols + j] - max_val).exp();
                    result_data[i * cols + j] = exp_val;
                    sum = sum + exp_val;
                }
                
                // Normalize
                for j in 0..cols {
                    result_data[i * cols + j] = result_data[i * cols + j] / sum;
                }
            }
            
            Ok(Self {
                data: result_data,
                shape: self.shape.clone(),
            })
        } else {
            Err(StorageError::InvalidOperation(
                "softmax not yet implemented for this tensor shape/axis combination".to_string(),
            ))
        }
    }
}

// Conversion traits implementation
impl<T: DataType> ToDense<T> for EnhancedDenseStorage<T> {
    type Output = Self;
    
    fn to_dense(&self) -> crate::Result<Self::Output> {
        Ok(self.clone())
    }
}

impl<T: DataType> fmt::Display for EnhancedDenseStorage<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseStorage(shape={:?}, data=[", self.shape.dims())?;
        
        let max_display = 10;
        for (i, item) in self.data.iter().take(max_display).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        
        if self.data.len() > max_display {
            write!(f, ", ... ({} more)", self.data.len() - max_display)?;
        }
        
        write!(f, "])")
    }
}