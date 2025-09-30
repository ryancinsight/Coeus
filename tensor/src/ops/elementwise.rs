//! Element-wise arithmetic operations implementation
//!
//! This module contains the implementation of element-wise operations
//! with broadcasting support for the unified tensor architecture.

use crate::{Result, Tensor, TensorError};
use coeus_backend::{Backend, CpuBackend};
use rayon::prelude::*;
use std::sync::Arc;
use std::borrow::Cow;
use coeus_backend::BackendData;
use coeus_storage::TensorStorage;

/// Compute broadcast shape following NumPy/PyTorch broadcasting rules
fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);

    let mut result_shape = Vec::with_capacity(max_len);

    // Pad shorter shape with leading dimensions of size 1
    let padded_shape1 = if len1 < max_len {
        let padding = vec![1; max_len - len1];
        [padding.as_slice(), shape1].concat()
    } else {
        shape1.to_vec()
    };

    let padded_shape2 = if len2 < max_len {
        let padding = vec![1; max_len - len2];
        [padding.as_slice(), shape2].concat()
    } else {
        shape2.to_vec()
    };

    // Compute broadcast shape
    for (dim1, dim2) in padded_shape1.iter().zip(padded_shape2.iter()) {
        if *dim1 == *dim2 {
            result_shape.push(*dim1);
        } else if *dim1 == 1 {
            result_shape.push(*dim2);
        } else if *dim2 == 1 {
            result_shape.push(*dim1);
        } else {
            return Err(crate::TensorError::BroadcastingError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }
    }

    Ok(result_shape)
}

/// Element-wise addition of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise sum or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Add};
///
/// let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
/// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
///
/// let result = a.add(&b).unwrap();
/// assert_eq!(result.data(), &[4.0, 6.0]);
/// ```
pub fn add<T: crate::Dtype + std::ops::Add<Output = T> + num_traits::NumCast, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B, S>,
    other: &Tensor<T, B, S>,
) -> Result<Tensor<T, B, S>>
where
    CpuBackend: Backend<T>,
{
    // Try advanced broadcasting first
    match compute_broadcast_shape(&tensor.shape, &other.shape) {
        Ok(result_shape) => {
            // Broadcast both tensors to result shape
            let tensor_data_vec = if tensor.shape == result_shape {
                tensor.data().to_vec()
            } else {
                let broadcasted = broadcast_data(&tensor.data, &tensor.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            let other_data_vec = if other.shape == result_shape {
                other.data().to_vec()
            } else {
                let broadcasted = broadcast_data(&other.data, &other.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            // Use backend for addition operation
            let backend_tensor_a = tensor.backend.create_tensor_data(tensor_data_vec, result_shape.clone())?;
            let backend_tensor_b = tensor.backend.create_tensor_data(other_data_vec, result_shape.clone())?;

            let backend_tensor_result = tensor.backend.add(&backend_tensor_a, &backend_tensor_b)?;

            // Convert backend tensor data to tensor crate tensor data
            let tensor_data = crate::TensorData {
                shape: backend_tensor_result.shape.clone(),
                data: Cow::Owned(backend_tensor_result.data.data.clone()),
            };
            let mut result = Tensor::from_backend_data(tensor.backend.clone(), Arc::new(tensor_data), result_shape);

            // Set requires_grad if either input requires gradients
            if false /* tensor.requires_grad() || other.requires_grad() */ {
                // result.set_node(coeus_autograd::graph::NodeId(0)); // DISABLED - autograd architectural redesign required
            }

            Ok(result)
        }
        Err(_) => {
            // Fallback to original simple broadcasting for backward compatibility
            let (result_shape, tensor_data_vec, other_data_vec) = if tensor.shape == other.shape {
                // Same shape - direct addition
                (
                    tensor.shape.clone(),
                    tensor.data().to_vec(),
                    other.data().to_vec(),
                )
            } else if tensor.shape.is_empty() && !other.shape.is_empty() {
                // Broadcast scalar tensor to match other's shape
                let broadcast_value = tensor.data()[0];
                let broadcast_data = vec![broadcast_value; other.numel()];
                (other.shape.clone(), broadcast_data, other.data().to_vec())
            } else if other.shape.is_empty() && !tensor.shape.is_empty() {
                // Broadcast scalar other to match tensor's shape
                let broadcast_value = other.data()[0];
                let broadcast_data = vec![broadcast_value; tensor.numel()];
                (tensor.shape.clone(), tensor.data().to_vec(), broadcast_data)
            } else {
                return Err(crate::TensorError::BroadcastingError {
                    shape1: tensor.shape.to_vec(),
                    shape2: other.shape.to_vec(),
                });
            };

            // Use parallel processing for large tensors to improve performance
            let data: Vec<T> = if tensor_data_vec.len() > 1000 {
                tensor_data_vec
                    .par_iter()
                    .zip(&other_data_vec)
                    .map(|(a, b)| *a + *b)
                    .collect()
            } else {
                tensor_data_vec
                    .iter()
                    .zip(&other_data_vec)
                    .map(|(a, b)| *a + *b)
                    .collect()
            };

            // Create backend tensor first
            let backend = tensor.backend.clone();
            let backend_result = Tensor::from_vec(backend, data, result_shape).map_err(|e| {
                TensorError::InvalidOperation {
                    message: format!("Failed to create result tensor: {}", e),
                }
            })?;

            // backend_result is already a valid tensor, so we can use it directly
            let mut result = backend_result;

            // Set requires_grad if either input requires gradients
            if false /* tensor.requires_grad() || other.requires_grad() */ {
                // result.set_node(coeus_autograd::graph::NodeId(0)); // DISABLED - autograd architectural redesign required
            }

            Ok(result)
        }
    }
}

/// Element-wise subtraction of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise difference or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Sub};
///
/// let a = Tensor::from_vec(vec![5.0, 7.0], vec![2]);
/// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
///
/// let result = a.sub(&b).unwrap();
/// assert_eq!(result.data(), &[2.0, 3.0]);
/// ```
pub fn sub<T: crate::Dtype + std::ops::Sub<Output = T> + crate::Dtype + num_traits::NumCast, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B, S>,
    other: &Tensor<T, B, S>,
) -> Result<Tensor<T, B, S>>
where
    CpuBackend: Backend<T>,
{
    // Try advanced broadcasting first
    match compute_broadcast_shape(&tensor.shape, &other.shape) {
        Ok(result_shape) => {
            // Broadcast both tensors to result shape
            let tensor_data_vec = if tensor.shape == result_shape {
                tensor.data().to_vec()
            } else {
                broadcast_data(&tensor.data, &tensor.shape, &result_shape)
            };

            let other_data_vec = if other.shape == result_shape {
                other.data().to_vec()
            } else {
                broadcast_data(&other.data, &other.shape, &result_shape)
            };

            // Use backend for subtraction operation
            let backend_tensor_a = tensor.backend.create_tensor_data(tensor_data_vec, result_shape.clone())?;
            let backend_tensor_b = tensor.backend.create_tensor_data(other_data_vec, result_shape.clone())?;

            let backend_result = tensor.backend.sub(&backend_tensor_a, &backend_tensor_b)?;

            // Extract result data
            let data = match &backend_result.data.data {
                coeus_backend::BackendData::Cpu(result_data) => result_data.clone(),
                coeus_backend::BackendData::Gpu(_) => return Err(TensorError::DeviceError { device: "GPU".to_string() }),
            };

            let result = Tensor::from_vec(tensor.backend.clone(), data, result_shape.clone())?;

            // TODO: Re-enable autograd integration after fixing type/API mismatches

            Ok(result)
        }
        Err(_) => Err(TensorError::ShapeMismatch {
            expected: tensor.shape.clone(),
            actual: other.shape.clone(),
        }),
    }
}

/// Element-wise multiplication of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise product or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Mul};
///
/// let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
/// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);
///
/// let result = a.mul(b).unwrap();
/// assert_eq!(result.data(), &[8.0, 15.0]);
/// ```
pub fn mul<T: crate::Dtype + std::ops::Mul<Output = T> + crate::Dtype + num_traits::NumCast, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B, S>,
    other: &Tensor<T, B, S>,
) -> Result<Tensor<T, B, S>>
where
    CpuBackend: Backend<T>,
{
    // Try advanced broadcasting first
    match compute_broadcast_shape(&tensor.shape, &other.shape) {
        Ok(result_shape) => {
            // Broadcast both tensors to result shape
            let tensor_data = if tensor.shape == result_shape {
                tensor.data.clone()
            } else {
                let broadcasted = broadcast_data(&tensor.data, &tensor.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            let other_data = if other.shape == result_shape {
                other.data.clone()
            } else {
                let broadcasted = broadcast_data(&other.data, &other.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            // Use backend for multiplication operation
            let backend_tensor_a = tensor.backend.create_tensor_data(tensor_data.data(), result_shape.clone())?;
            let backend_tensor_b = tensor.backend.create_tensor_data(other_data.data(), result_shape.clone())?;

            let backend_result = tensor.backend.mul(&backend_tensor_a, &backend_tensor_b)?;

            // Extract result data
            let data = match &backend_result.data.data {
                coeus_backend::BackendData::Cpu(result_data) => result_data.clone(),
                coeus_backend::BackendData::Gpu(_) => return Err(TensorError::DeviceError { device: "GPU".to_string() }),
            };

            // Create backend tensor first
            let backend = tensor.backend.clone();
            let backend_result = Tensor::from_vec(backend, data, result_shape).map_err(|e| {
                TensorError::InvalidOperation {
                    message: format!("Failed to create result tensor: {}", e),
                }
            })?;

            // backend_result is already a valid tensor, so we can use it directly
            let mut result = backend_result;

    // TODO: Re-enable autograd integration after fixing autograd crate

    Ok(result)
        }
        Err(_) => Err(TensorError::ShapeMismatch {
            expected: tensor.shape.clone(),
            actual: other.shape.clone(),
        }),
    }
}

/// Element-wise division of tensors
///
/// # Arguments
/// * `self` - First tensor (mutable for potential gradient tracking)
/// * `other` - Second tensor (mutable for potential gradient tracking)
///
/// # Returns
/// Result containing the element-wise quotient or an error
///
/// # Errors
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Div};
///
/// let a = Tensor::from_vec(vec![8.0, 15.0], vec![2]);
/// let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);
///
/// let result = a.div(&b).unwrap();
/// assert_eq!(result.data(), &[2.0, 3.0]);
/// ```
pub fn div<T: crate::Dtype + std::ops::Div<Output = T> + crate::Dtype + num_traits::NumCast, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B, S>,
    other: &Tensor<T, B, S>,
) -> Result<Tensor<T, B, S>>
where
    CpuBackend: Backend<T>,
{
    // Try advanced broadcasting first
    match compute_broadcast_shape(&tensor.shape, &other.shape) {
        Ok(result_shape) => {
            // Broadcast both tensors to result shape
            let tensor_data = if tensor.shape == result_shape {
                tensor.data.clone()
            } else {
                let broadcasted = broadcast_data(&tensor.data, &tensor.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            let other_data = if other.shape == result_shape {
                other.data.clone()
            } else {
                let broadcasted = broadcast_data(&other.data, &other.shape, &result_shape);
                Arc::new(crate::TensorData::new_cpu(broadcasted, result_shape.clone()))
            };

            // Use backend for division operation
            let backend_tensor_a = tensor.backend.create_tensor_data(tensor_data.data(), result_shape.clone())?;
            let backend_tensor_b = tensor.backend.create_tensor_data(other_data.data(), result_shape.clone())?;

            let backend_result = tensor.backend.div(&backend_tensor_a, &backend_tensor_b)?;

            // Extract result data
            let data = match &backend_result.data.data {
                coeus_backend::BackendData::Cpu(result_data) => result_data.clone(),
                coeus_backend::BackendData::Gpu(_) => return Err(TensorError::DeviceError { device: "GPU".to_string() }),
            };

            // Create backend tensor first
            let backend = tensor.backend.clone();
            let backend_result = Tensor::from_vec(backend, data, result_shape).map_err(|e| {
                TensorError::InvalidOperation {
                    message: format!("Failed to create result tensor: {}", e),
                }
            })?;

            // backend_result is already a valid tensor, so we can use it directly
            let mut result = backend_result;

    // TODO: Re-enable autograd integration after fixing autograd crate

    Ok(result)
        }
        Err(_) => Err(TensorError::ShapeMismatch {
            expected: tensor.shape.clone(),
            actual: other.shape.clone(),
        }),
    }
}

/// Element-wise negation of a tensor
///
/// # Arguments
/// * `tensor` - Tensor to negate (mutable for potential gradient tracking)
///
/// # Returns
/// The element-wise negation of the tensor
///
/// # Example
/// ```rust
/// use coeus_tensor::{Tensor, Neg};
///
/// let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
/// let result = a.neg();
/// assert_eq!(result.data(), &[-1.0, 2.0, -3.0]);
/// ```
pub fn neg<T: crate::Dtype + std::ops::Neg<Output = T> + Clone, B: Backend<T> + Clone, S: TensorStorage<T> + Clone + Send + Sync>(
    tensor: &Tensor<T, B, S>,
) -> Result<Tensor<T, B, S>>
where
    CpuBackend: Backend<T>,
{
    let data = tensor.data().iter().map(|x| -*x).collect();

    // Create result tensor with same backend and autograd settings
    let mut result = Tensor::from_vec(tensor.backend.clone(), data, tensor.shape.clone())?;

    // TODO: Re-enable autograd integration after fixing autograd crate

    Ok(result)
}

// Operator overload implementations moved to core/tensor.rs to avoid conflicts

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let result = add(&a, &b).unwrap();
        assert_eq!(result.data, &[4.0, 6.0]);
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::from_vec(vec![5.0, 7.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let result = sub(&a, &b).unwrap();
        assert_eq!(result.data, &[2.0, 3.0]);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);

        let result = mul(&a, &b).unwrap();
        assert_eq!(result.data, &[8.0, 15.0]);
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::from_vec(vec![8.0, 15.0], vec![2]);
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]);

        let result = div(&a, &b).unwrap();
        assert_eq!(result.data, &[2.0, 3.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], vec![3]);
        let result = neg(&a);
        assert_eq!(result.data, &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_operator_overloads() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        // Test addition
        let result = (&a + &b).unwrap();
        assert_eq!(result.data, &[4.0, 6.0]);

        // Test subtraction
        let result = (&a - &b).unwrap();
        assert_eq!(result.data, &[-2.0, -2.0]);

        // Test multiplication
        let result = (&a * &b).unwrap();
        assert_eq!(result.data, &[3.0, 8.0]);

        // Test division
        let result = (&b / &a).unwrap();
        assert_eq!(result.data, &[3.0, 2.0]);

        // Test negation
        let result = -&a;
        assert_eq!(result.data, &[-1.0, -2.0]);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);

        // Test addition with different shapes
        let result = &a + &b;
        assert!(result.is_err());

        // Test subtraction with different shapes
        let result = &a - &b;
        assert!(result.is_err());

        // Test multiplication with different shapes
        let result = &a * &b;
        assert!(result.is_err());

        // Test division with different shapes
        let result = &a / &b;
        assert!(result.is_err());
    }
}
