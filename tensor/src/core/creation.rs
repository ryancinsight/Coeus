//! Tensor creation and initialization operations
//!
//! This module contains methods for creating tensors with various initializations,
//! including zeros, ones, identity matrices, and scalar tensors.

use crate::{Device, Layout, Tensor, TensorError, Dtype, FloatDtype, Result};

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Create a tensor from a vector and shape
    ///
    /// # Arguments
    /// * `data` - Vector containing tensor elements in row-major order
    /// * `shape` - Shape of the tensor
    ///
    /// # Errors
    /// Returns `TensorError::InvalidShape` if the data length doesn't match the shape product
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();
    /// ```
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> crate::Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(crate::TensorError::InvalidShape {
                data_len: data.len(),
                shape_product: expected_len,
                shape: shape.clone(),
            });
        }

        Ok(Tensor {
            data,
            shape,
            device: Device::Cpu,
            layout: Layout::default(),
            node: None,
            context: None,
            grad: std::sync::Arc::new(std::sync::RwLock::new(None)),
            input_tensor_nodes: vec![],
        })
    }

    /// Create a tensor from a vector and shape with gradient tracking enabled
    ///
    /// # Arguments
    /// * `data` - Vector containing tensor elements
    /// * `shape` - Shape of the tensor
    ///
    /// # Returns
    /// A Result containing the tensor with gradient tracking enabled or a TensorError
    ///
    /// # Errors
    /// Returns `TensorError::InvalidShape` if the data length doesn't match the shape product
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let tensor = Tensor::from_vec_with_grad(data, vec![3]).unwrap();
    /// ```
    pub fn from_vec_with_grad(data: Vec<T>, shape: Vec<usize>) -> crate::Result<Self>
    where
        T: FloatDtype + std::ops::Neg<Output = T>,
    {
        let mut tensor = Self::from_vec(data, shape)?;
        tensor.set_requires_grad(true);
        Ok(tensor)
    }

    /// Create a tensor filled with zeros
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let zeros = Tensor::<f32>::zeros(vec![2, 3]);
    /// assert_eq!(zeros.shape(), &[2, 3]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Self::from_vec(data, shape)
    }

    /// Create a tensor filled with ones
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let ones = Tensor::<f32>::ones(vec![2, 3]);
    /// assert_eq!(ones.shape(), &[2, 3]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Self::from_vec(data, shape)
    }

    /// Create an identity matrix
    ///
    /// # Arguments
    /// * `size` - Size of the square matrix
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let eye = Tensor::<f32>::eye(3);
    /// assert_eq!(eye.shape(), &[3, 3]);
    /// ```
    pub fn eye(size: usize) -> Self {
        let mut data = vec![T::zero(); size * size];
        for i in 0..size {
            data[i * size + i] = T::one();
        }
        Self::from_vec(data, vec![size, size])
    }

    /// Create a scalar tensor
    ///
    /// # Arguments
    /// * `value` - Scalar value
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let scalar = Tensor::<f32>::scalar(3.14);
    /// assert_eq!(scalar.shape(), &[]);
    /// ```
    pub fn scalar(value: T) -> Self {
        Self::from_vec(vec![value], vec![])
    }

    /// Create a tensor from data with explicit device specification
    ///
    /// # Arguments
    /// * `data` - Vector containing tensor elements
    /// * `shape` - Shape of the tensor
    /// * `device` - Device to place the tensor on
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::{Tensor, Device};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_vec_device(data, vec![2, 2], Device::Cpu);
    /// ```
    pub fn from_vec_device(data: Vec<T>, shape: Vec<usize>, device: Device) -> Self {
        let mut tensor = Self::from_vec(data, shape);
        tensor.device = device;
        tensor
    }

    /// Create a tensor filled with zeros with the same shape as another tensor
    ///
    /// # Arguments
    /// * `other` - Tensor to match the shape of
    ///
    /// # Example
    /// ```
    /// use coeus_tensor::Tensor;
    ///
    /// let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// let zeros = Tensor::zeros_like(&a);
    /// assert_eq!(zeros.shape(), a.shape());
    /// ```
    pub fn zeros_like(other: &Tensor<T>) -> Tensor<T> {
        Tensor::zeros(other.shape.clone())
    }
}
