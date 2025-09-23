//! Tensor reference types for the computational graph

use coeus_dtype::Dtype;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Reference to a tensor with shape information
#[derive(Clone)]
pub struct TensorRef<T: Dtype> {
    /// Flat data storage
    data: Vec<T>,
    /// Shape of the tensor
    shape: Vec<usize>,
}

impl<T: Dtype> TensorRef<T> {
    /// Create a scalar tensor
    pub fn scalar(value: T) -> Self {
        Self {
            data: vec![value],
            shape: vec![1],
        }
    }

    /// Create a tensor from data and shape
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape product"
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![T::zero(); size],
            shape,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![T::one(); size],
            shape,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get element at flat index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get mutable element at flat index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    /// Check if tensor is scalar
    pub fn is_scalar(&self) -> bool {
        self.shape == [1]
    }

    /// Get scalar value (panics if not scalar)
    pub fn as_scalar(&self) -> T {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        self.data[0]
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a + *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch in subtraction");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a - *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch in multiplication");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise division
    pub fn div(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "Shape mismatch in division");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a / *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise negation
    pub fn neg(&self) -> Self
    where
        T: std::ops::Neg<Output = T>,
    {
        let data = self.data.iter().map(|x| -*x).collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Sum of all elements
    pub fn sum(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, x| acc + *x)
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert_eq!(
            self.numel(),
            new_shape.iter().product::<usize>(),
            "Cannot reshape: size mismatch"
        );
        Self {
            data: self.data.clone(),
            shape: new_shape,
        }
    }

    /// Transpose (for 2D tensors)
    pub fn t(&self) -> Self {
        assert_eq!(self.ndim(), 2, "Transpose only supported for 2D tensors");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut new_data = vec![T::zero(); self.numel()];

        for i in 0..rows {
            for j in 0..cols {
                let old_idx = i * cols + j;
                let new_idx = j * rows + i;
                new_data[new_idx] = self.data[old_idx];
            }
        }

        Self {
            data: new_data,
            shape: vec![cols, rows],
        }
    }
}

impl<T: Dtype> Add for TensorRef<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::add(&self, &other)
    }
}

impl<T: Dtype> Sub for TensorRef<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::sub(&self, &other)
    }
}

impl<T: Dtype> Mul for TensorRef<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self::mul(&self, &other)
    }
}

impl<T: Dtype> Div for TensorRef<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self::div(&self, &other)
    }
}

impl<T: Dtype + std::ops::Neg<Output = T>> Neg for TensorRef<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(&self)
    }
}

impl<T: Dtype> PartialEq for TensorRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T: Dtype> fmt::Debug for TensorRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorRef {{ shape: {:?}, data: {:?} }}",
            self.shape, self.data
        )
    }
}

impl<T: Dtype + fmt::Display> fmt::Display for TensorRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            write!(f, "{}", self.as_scalar())
        } else {
            write!(f, "Tensor(shape={:?})", self.shape)
        }
    }
}

/// Iterator over tensor elements
pub struct TensorIter<'a, T: Dtype> {
    tensor: &'a TensorRef<T>,
    index: usize,
}

impl<'a, T: Dtype> Iterator for TensorIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.tensor.numel() {
            let item = &self.tensor.data[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<T: Dtype> TensorRef<T> {
    /// Create an iterator over tensor elements
    pub fn iter(&self) -> TensorIter<'_, T> {
        TensorIter {
            tensor: self,
            index: 0,
        }
    }
}

/// Mutable iterator over tensor elements
pub struct TensorIterMut<'a, T: Dtype> {
    data: &'a mut [T],
    index: usize,
}

impl<'a, T: Dtype> Iterator for TensorIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            // Safe mutable access: we have exclusive access to self.data and are returning
            // a reference with lifetime 'a which matches the iterator's lifetime
            let current_index = self.index;
            self.index += 1;
            // SAFETY: We have exclusive access to self.data and are returning a reference
            // with lifetime 'a which matches the iterator's lifetime parameter
            Some(unsafe { &mut *self.data.as_mut_ptr().add(current_index) })
        } else {
            None
        }
    }
}

impl<T: Dtype> TensorRef<T> {
    /// Create a mutable iterator over tensor elements
    pub fn iter_mut(&mut self) -> TensorIterMut<'_, T> {
        TensorIterMut {
            data: &mut self.data,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_creation() {
        let t = TensorRef::scalar(5.0f32);
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.as_scalar(), 5.0);
    }

    #[test]
    fn test_tensor_operations() {
        let a = TensorRef::from_data(vec![1.0, 2.0], vec![2]);
        let b = TensorRef::from_data(vec![3.0, 4.0], vec![2]);
        let c = a.add(b);
        assert_eq!(c.data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_iterator() {
        let t = TensorRef::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let sum: f32 = t.iter().sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn test_reshape() {
        let t = TensorRef::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let reshaped = t.reshape(vec![2, 2]);
        assert_eq!(reshaped.shape(), &[2, 2]);
    }
}
