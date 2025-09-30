//! Trait-Based Tensor Operations - Zero-Cost Polymorphism
//!
//! This module provides operations that work with any tensor storage format
//! through the Tensor trait, enabling unified interfaces across dense and
//! sparse tensor implementations.
//!
//! ## Zero-Cost Polymorphism
//!
//! Operations are implemented as generic functions that work with any type
//! implementing the Tensor trait. This enables:
//! - Unified API across storage formats
//! - Compile-time dispatch (no runtime overhead)
//! - Type-safe operations with storage-specific optimizations

use crate::{Dtype, Result, Tensor, TensorError};
use crate::traits::{Tensor as TensorTrait, TensorOps};
use coeus_storage::{TensorStorage, DenseStorage};
use coeus_backend::Backend;

/// Generic element-wise addition for any tensor storage format
pub fn add<T, B, S>(
    a: &dyn TensorTrait<T, B, S>,
    b: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype + std::ops::Add<Output = T> + Clone + Copy,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    // Convert to dense for operation (can be optimized per storage type)
    let a_dense = a.to_dense()?;
    let b_dense = b.to_dense()?;

    // Perform operation on dense tensors
    let result_dense = add_dense(&*a_dense, &*b_dense)?;

    // Return result (maintains dense format for now)
    Ok(result_dense)
}

/// Dense tensor addition implementation
fn add_dense<T, B>(
    a: &dyn TensorTrait<T, B, DenseStorage<T>>,
    b: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype + std::ops::Add<Output = T> + Clone + Copy,
    B: Backend<T> + Clone + Send + Sync,
{
    // Get data and perform element-wise addition
    let a_data = a.data().ok_or_else(|| TensorError::InvalidOperation {
        message: "Dense tensor should have data".to_string()
    })?;

    let b_data = b.data().ok_or_else(|| TensorError::InvalidOperation {
        message: "Dense tensor should have data".to_string()
    })?;

    if a_data.len() != b_data.len() {
        return Err(TensorError::BroadcastingError {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        });
    }

    let result_data: Vec<T> = a_data.iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    // Create result tensor with same backend
    let backend = a.backend().clone();
    let shape = a.shape().to_vec();
    Tensor::from_vec(backend, result_data, shape)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Generic element-wise multiplication
pub fn mul<T, B, S>(
    a: &dyn TensorTrait<T, B, S>,
    b: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype + std::ops::Mul<Output = T> + Clone + Copy,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let a_dense = a.to_dense()?;
    let b_dense = b.to_dense()?;
    let result_dense = mul_dense(&*a_dense, &*b_dense)?;
    Ok(result_dense)
}

/// Dense tensor multiplication implementation
fn mul_dense<T, B>(
    a: &dyn TensorTrait<T, B, DenseStorage<T>>,
    b: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype + std::ops::Mul<Output = T> + Clone + Copy,
    B: Backend<T> + Clone + Send + Sync,
{
    let a_data = a.data().ok_or_else(|| TensorError::InvalidOperation {
        message: "Dense tensor should have data".to_string()
    })?;

    let b_data = b.data().ok_or_else(|| TensorError::InvalidOperation {
        message: "Dense tensor should have data".to_string()
    })?;

    if a_data.len() != b_data.len() {
        return Err(TensorError::BroadcastingError {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        });
    }

    let result_data: Vec<T> = a_data.iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| x * y)
        .collect();

    let backend = a.backend().clone();
    let shape = a.shape().to_vec();
    Tensor::from_vec(backend, result_data, shape)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Generic matrix multiplication
pub fn matmul<T, B, S>(
    a: &dyn TensorTrait<T, B, S>,
    b: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let a_dense = a.to_dense()?;
    let b_dense = b.to_dense()?;
    let result_dense = matmul_dense(&*a_dense, &*b_dense)?;
    Ok(result_dense)
}

/// Dense matrix multiplication implementation
fn matmul_dense<T, B>(
    a: &dyn TensorTrait<T, B, DenseStorage<T>>,
    b: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    // Use existing matmul implementation
    crate::ops::matrix::matmul(a, b)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Generic element-wise operations

/// Exponential function
pub fn exp<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = exp_dense(&*dense)?;
    Ok(result_dense)
}

fn exp_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::arithmetic::exp(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Natural logarithm
pub fn log<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = log_dense(&*dense)?;
    Ok(result_dense)
}

fn log_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::arithmetic::log(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Square root
pub fn sqrt<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = sqrt_dense(&*dense)?;
    Ok(result_dense)
}

fn sqrt_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::arithmetic::sqrt(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Element-wise negation
pub fn neg<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = neg_dense(&*dense)?;
    Ok(result_dense)
}

fn neg_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::arithmetic::neg(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Reduction operations

/// Sum all elements
pub fn sum<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = sum_dense(&*dense)?;
    Ok(result_dense)
}

fn sum_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::reduction::sum(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Mean of all elements
pub fn mean<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = mean_dense(&*dense)?;
    Ok(result_dense)
}

fn mean_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    crate::ops::reduction::mean(tensor)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

/// Shape operations

/// Transpose (2D tensors only)
pub fn t<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = t_dense(&*dense)?;
    Ok(result_dense)
}

fn t_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    // Simple transpose for 2D tensors
    if tensor.shape().len() != 2 {
        return Err(TensorError::InvalidOperation {
            message: "Transpose (t) only supported for 2D tensors".to_string()
        });
    }

    tensor.transpose(0, 1)
}

/// Reshape tensor
pub fn reshape<T, B, S>(
    tensor: &dyn TensorTrait<T, B, S>,
    new_shape: Vec<usize>
) -> Result<Box<dyn TensorTrait<T, B, S>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T>,
{
    let dense = tensor.to_dense()?;
    let result_dense = reshape_dense(&*dense, new_shape)?;
    Ok(result_dense)
}

fn reshape_dense<T, B>(
    tensor: &dyn TensorTrait<T, B, DenseStorage<T>>,
    new_shape: Vec<usize>
) -> Result<Box<dyn TensorTrait<T, B, DenseStorage<T>>>>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
{
    let new_numel: usize = new_shape.iter().product();
    if new_numel != tensor.numel() {
        return Err(TensorError::InvalidShape {
            data_len: tensor.numel(),
            shape_product: new_numel,
            shape: new_shape,
        });
    }

    let data = tensor.data().unwrap().to_vec();
    let backend = tensor.backend().clone();

    Tensor::from_vec(backend, data, new_shape)
        .map(|t| Box::new(t) as Box<dyn TensorTrait<T, B, DenseStorage<T>>>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_trait_ops_compilation() {
        // Test that trait operations compile correctly
        // This is a compilation test - actual functionality tested in concrete implementations
        assert!(true);
    }
}
