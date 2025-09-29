//! Tensor creation and initialization operations
//!
//! This module contains functions for creating tensors with various initializations,
//! including zeros, ones, identity matrices, and scalar tensors for the unified architecture.

use crate::{Dtype, FloatDtype, Result, Tensor, TensorError};
use coeus_backend::Backend;

/// Create a tensor from a vector and shape using a specific backend
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `data` - Vector containing tensor elements in row-major order
/// * `shape` - Shape of the tensor
///
/// # Errors
/// Returns `TensorError::InvalidShape` if the data length doesn't match the shape product
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = Tensor::from_vec(backend, data, vec![2, 2]).unwrap();
/// ```
pub fn from_vec<T: Dtype, B: Backend<T> + Clone>(
    backend: B,
    data: Vec<T>,
    shape: Vec<usize>
) -> Result<Tensor<T, B>> {
    let expected_len: usize = shape.iter().product();
    if data.len() != expected_len {
        return Err(TensorError::InvalidShape {
            data_len: data.len(),
            shape_product: expected_len,
            shape: shape.clone(),
        });
    }

    Ok(Tensor::from_vec(backend, data, shape)?)
}

/// Create a tensor from a vector and shape with gradient tracking enabled
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
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
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let data = vec![1.0, 2.0, 3.0];
/// let tensor = Tensor::from_vec_with_grad(backend, data, vec![3]).unwrap();
/// ```
pub fn from_vec_with_grad<T: FloatDtype, B: Backend<T> + Clone>(
    backend: B,
    data: Vec<T>,
    shape: Vec<usize>
) -> Result<Tensor<T, B>> {
    let mut tensor = from_vec(backend, data, shape)?;
    tensor.set_requires_grad(true);
    Ok(tensor)
}

/// Create a tensor filled with zeros
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `shape` - Shape of the tensor
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let zeros = Tensor::zeros(backend, vec![2, 3]).unwrap();
/// assert_eq!(zeros.shape(), &[2, 3]);
/// ```
pub fn zeros<T: Dtype, B: Backend<T> + Clone>(backend: B, shape: Vec<usize>) -> Result<Tensor<T, B>> {
    let numel = shape.iter().product();
    let data = vec![T::zero(); numel];
    from_vec(backend, data, shape)
}

/// Create a tensor filled with ones
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `shape` - Shape of the tensor
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let ones = Tensor::ones(backend, vec![2, 3]).unwrap();
/// ```
pub fn ones<T: Dtype, B: Backend<T> + Clone>(backend: B, shape: Vec<usize>) -> Result<Tensor<T, B>> {
    let numel = shape.iter().product();
    let data = vec![T::one(); numel];
    from_vec(backend, data, shape)
}

/// Create a scalar tensor
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `value` - Scalar value
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let scalar = Tensor::scalar(backend, 3.14).unwrap();
/// assert_eq!(scalar.shape(), &[]);
/// ```
pub fn scalar<T: Dtype, B: Backend<T> + Clone>(backend: B, value: T) -> Result<Tensor<T, B>> {
    from_vec(backend, vec![value], vec![])
}

/// Create a tensor filled with a specific value
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `shape` - Shape of the tensor
/// * `value` - Value to fill the tensor with
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let filled = Tensor::full(backend, vec![2, 2], 5.0).unwrap();
/// ```
pub fn full<T: Dtype, B: Backend<T> + Clone>(backend: B, shape: Vec<usize>, value: T) -> Result<Tensor<T, B>> {
    let numel = shape.iter().product();
    let data = vec![value; numel];
    from_vec(backend, data, shape)
}

/// Create an identity matrix tensor
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `n` - Size of the square matrix
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let identity = Tensor::eye(backend, 3).unwrap();
/// ```
pub fn eye<T: Dtype + num_traits::FromPrimitive, B: Backend<T> + Clone>(
    backend: B,
    n: usize
) -> Result<Tensor<T, B>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    from_vec(backend, data, vec![n, n])
}

/// Create a tensor with values from 0 to n-1
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `n` - Number of elements
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let arange = Tensor::arange(backend, 5).unwrap();
/// // Creates [0, 1, 2, 3, 4]
/// ```
pub fn arange<T: Dtype + num_traits::FromPrimitive, B: Backend<T> + Clone>(
    backend: B,
    n: usize
) -> Result<Tensor<T, B>> {
    let data: Vec<T> = (0..n).map(|i| T::from_usize(i).unwrap_or(T::zero())).collect();
    from_vec(backend, data, vec![n])
}

/// Create a tensor with linearly spaced values
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `start` - Starting value
/// * `end` - Ending value
/// * `steps` - Number of steps
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let linspace = Tensor::linspace(backend, 0.0, 1.0, 5).unwrap();
/// // Creates [0.0, 0.25, 0.5, 0.75, 1.0]
/// ```
pub fn linspace<T: FloatDtype, B: Backend<T> + Clone>(
    backend: B,
    start: T,
    end: T,
    steps: usize
) -> Result<Tensor<T, B>> {
    if steps == 0 {
        return from_vec(backend, vec![], vec![0]);
    }

    let step_size = if steps == 1 {
        T::zero()
    } else {
        (end - start) / T::from_usize(steps - 1).unwrap_or(T::one())
    };

    let data: Vec<T> = (0..steps)
        .map(|i| start + step_size * T::from_usize(i).unwrap_or(T::zero()))
        .collect();

    from_vec(backend, data, vec![steps])
}

/// Create a tensor with logarithmically spaced values
///
/// # Arguments
/// * `backend` - Backend to use for tensor operations
/// * `start` - Starting value (base^start)
/// * `end` - Ending value (base^end)
/// * `steps` - Number of steps
/// * `base` - Base of the logarithm (default: 10.0)
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, CpuBackend};
///
/// let backend = CpuBackend::new();
/// let logspace = Tensor::logspace(backend, 0.0, 2.0, 5, 10.0).unwrap();
/// // Creates [1.0, 10^(0.5), 10^1, 10^(1.5), 10^2]
/// ```
pub fn logspace<T: FloatDtype, B: Backend<T> + Clone>(
    backend: B,
    start: T,
    end: T,
    steps: usize,
    base: T
) -> Result<Tensor<T, B>> {
    let data: Vec<T> = (0..steps)
        .map(|i| {
            let ratio = T::from_usize(i).unwrap_or(T::zero()) / T::from_usize(steps - 1).unwrap_or(T::one());
            let exponent = start + (end - start) * ratio;
            base.powf(exponent)
        })
        .collect();

    from_vec(backend, data, vec![steps])
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;

    #[test]
    fn test_from_vec() {
        let backend = CpuBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = from_vec(backend, data, vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros() {
        let backend = CpuBackend::new();
        let tensor = zeros::<f32, CpuBackend>(backend, vec![2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert!(tensor.data().iter().all(|&x: &f32| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let backend = CpuBackend::new();
        let tensor = ones::<f32, CpuBackend>(backend, vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert!(tensor.data().iter().all(|&x: &f32| x == 1.0));
    }

    #[test]
    fn test_scalar() {
        let backend = CpuBackend::new();
        let tensor = scalar::<f64, CpuBackend>(backend, 3.14f64).unwrap();
        assert_eq!(tensor.shape(), &[] as &[usize]);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.data()[0], 3.14);
    }

    #[test]
    fn test_full() {
        let backend = CpuBackend::new();
        let tensor = full(backend, vec![2, 2], 5.0).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert!(tensor.data().iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_eye() {
        let backend = CpuBackend::new();
        let tensor = eye::<f32, CpuBackend>(backend, 3).unwrap();
        assert_eq!(tensor.shape(), &[3, 3]);

        // Check diagonal is 1s
        assert_eq!(tensor.data()[0], 1.0); // (0,0)
        assert_eq!(tensor.data()[4], 1.0); // (1,1)
        assert_eq!(tensor.data()[8], 1.0); // (2,2)

        // Check off-diagonal is 0s
        assert_eq!(tensor.data()[1], 0.0); // (0,1)
        assert_eq!(tensor.data()[3], 0.0); // (1,0)
    }

    #[test]
    fn test_arange() {
        let backend = CpuBackend::new();
        let tensor = arange::<f32, CpuBackend>(backend, 5).unwrap();
        assert_eq!(tensor.shape(), &[5]);
        assert_eq!(tensor.data(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linspace() {
        let backend = CpuBackend::new();
        let tensor = linspace(backend, 0.0, 1.0, 5).unwrap();
        assert_eq!(tensor.shape(), &[5]);
        assert_eq!(tensor.data()[0], 0.0);
        assert_eq!(tensor.data()[1], 0.25);
        assert_eq!(tensor.data()[2], 0.5);
        assert_eq!(tensor.data()[3], 0.75);
        assert_eq!(tensor.data()[4], 1.0);
    }

    #[test]
    fn test_logspace() {
        let backend = CpuBackend::new();
        let tensor = logspace(backend, 0.0, 2.0, 5, 10.0).unwrap();
        assert_eq!(tensor.shape(), &[5]);
        assert_eq!(tensor.data()[0], 1.0);   // 10^0 = 1
        assert_eq!(tensor.data()[1], 10.0);  // 10^0.5 ≈ 3.16, but let's be approximate
        assert_eq!(tensor.data()[2], 100.0); // 10^1 = 100
        assert_eq!(tensor.data()[4], 10000.0); // 10^2 = 10000
    }

    #[test]
    fn test_from_vec_with_grad() {
        let backend = CpuBackend::new();
        let data = vec![1.0, 2.0, 3.0];
        let tensor = from_vec_with_grad(backend, data, vec![3]).unwrap();
        assert_eq!(tensor.shape(), &[3]);
        assert!(tensor.requires_grad());
    }
}
