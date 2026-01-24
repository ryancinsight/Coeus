//! Automatic differentiation tensor operations
//!
//! This module provides tensor operations that automatically construct the computation graph
//! for gradient computation. These functions mirror the operations in `tensor::arithmetic`
//! but attach `Function` objects to enable automatic differentiation.

extern crate alloc;

use crate::{
    functions::{
        AddFunction, CosFunction, DivFunction, ExpFunction, LogFunction, MatMulFunction,
        MaxFunction, MeanFunction, MulFunction, NegFunction, PowFunction, ReshapeFunction,
        SinFunction, SqrtFunction, SubFunction, SumFunction, TransposeFunction,
    },
    Result,
};
use alloc::sync::Arc;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::FromPrimitive;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;


// Function to create proper Function objects for gradient computation
fn create_add_function<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static,
{
    Arc::new(AddFunction::new(
        Arc::new(lhs.clone()),
        Arc::new(rhs.clone()),
    ))
}

fn create_sub_function<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(SubFunction::new(
        Arc::new(lhs.clone()),
        Arc::new(rhs.clone()),
    ))
}

fn create_matmul_function<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static,
{
    Arc::new(MatMulFunction::new(
        Arc::new(lhs.clone()),
        Arc::new(rhs.clone()),
    ))
}

fn create_mean_function<B, S, T>(
    input: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static + num_traits::FromPrimitive,
{
    let input_shape = input.shape().dims().to_vec();
    Arc::new(MeanFunction::new(Arc::new(input.clone()), input_shape))
}

fn create_sum_function<B, S, T>(
    input: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static,
{
    let input_shape = input.shape().dims().to_vec();
    Arc::new(SumFunction::new(Arc::new(input.clone()), input_shape))
}

fn create_mul_function<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + 'static + std::ops::Neg<Output = T>,
{
    Arc::new(MulFunction::new(
        Arc::new(lhs.clone()),
        Arc::new(rhs.clone()),
    ))
}

fn create_reshape_function<B, S, T>(
    input: &Tensor<B, S, T>,
    input_shape: Vec<usize>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static,
{
    Arc::new(ReshapeFunction::new(Arc::new(input.clone()), input_shape))
}

fn create_transpose_function<B, S, T>(
    input: &Tensor<B, S, T>,
    dim0: usize,
    dim1: usize,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + 'static,
{
    Arc::new(TransposeFunction::new(Arc::new(input.clone()), dim0, dim1))
}

fn create_div_function<B, S, T>(
    lhs: &Tensor<B, S, T>,
    rhs: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(DivFunction::new(
        Arc::new(lhs.clone()),
        Arc::new(rhs.clone()),
    ))
}

fn create_neg_function<B, S, T>(input: &Tensor<B, S, T>) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(NegFunction::new(Arc::new(input.clone())))
}

fn create_pow_function<B, S, T>(
    input: &Tensor<B, S, T>,
    exponent: f64,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + 'static,
{
    let exp_t = T::from_f64(exponent).expect("Failed to convert exponent to tensor data type");
    Arc::new(PowFunction::new(Arc::new(input.clone()), exp_t))
}

fn create_sqrt_function<B, S, T>(
    input: &Tensor<B, S, T>,
    output: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + 'static,
{
    Arc::new(SqrtFunction::new(Arc::new(input.clone()), Arc::new(output.clone())))
}

/// Element-wise square root with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn sqrt<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt + num_traits::FromPrimitive,
{
    // Perform sqrt operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let result_dense = input_dense.sqrt();

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let result_dims = result_dense.shape().dims();
    let result = Tensor::<B, S, T>::from_vec(data, result_dims)
        .map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        let grad_fn = create_sqrt_function(input, &result);
        Ok(result
            .with_grad_fn(Some(grad_fn))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise addition with automatic differentiation
///
/// This function performs element-wise addition and automatically attaches
/// an `AddFunction` to the result tensor if either input requires gradients.
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with automatic differentiation support
///
/// # Examples
///
/// ```rust
/// use tensor::{Tensor, CpuBackend, DenseStorage};
/// use dtype::float::Float32;
/// use autograd::tensor_ops::add;
///
/// let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(3.0), Float32::new(4.0)], &[2]
/// ).unwrap().requires_grad_(true);
///
/// let z = add(&x, &y).unwrap();
/// assert!(z.function_object().is_some()); // Has AddBackward function attached
/// ```
#[allow(clippy::missing_errors_doc)]
pub fn add<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Add<Output = T> + Copy + 'static,
{
    // Perform the addition operation using the arithmetic module directly to avoid trait ambiguity
    let result =
        tensor::ops::arithmetic::add(lhs, rhs).map_err(crate::AutogradError::TensorError)?;
    if tensor::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        Ok(result
            .with_grad_fn(Some(create_add_function(lhs, rhs)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Transpose operation with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn transpose<B, S, T>(
    input: &Tensor<B, S, T>,
    dim0: usize,
    dim1: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Clone + Copy + 'static,
{
    // Convert to dense for operation
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    // Perform transpose operation
    let result_dense = input_dense
        .transpose(dim0, dim1)
        .map_err(crate::AutogradError::TensorError)?;

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let result_dims = result_dense.shape().dims();
    let result = Tensor::<B, S, T>::from_vec(data, result_dims)
        .map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_transpose_function(input, dim0, dim1)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise multiplication with automatic differentiation
///
/// This function performs element-wise multiplication and automatically attaches
/// a `MulFunction` to the result tensor if either input requires gradients.
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with automatic differentiation support
#[allow(clippy::missing_errors_doc)]
pub fn mul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Mul<Output = T> + Copy + 'static + std::ops::Neg<Output = T>,
{
    // Perform the multiplication operation using the arithmetic module directly
    let result =
        tensor::ops::arithmetic::mul(lhs, rhs).map_err(crate::AutogradError::TensorError)?;
    if tensor::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        Ok(result
            .with_grad_fn(Some(create_mul_function(lhs, rhs)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise subtraction with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn sub<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Sub<Output = T> + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform the subtraction operation using the arithmetic module directly
    let result =
        tensor::ops::arithmetic::sub(lhs, rhs).map_err(crate::AutogradError::TensorError)?;
    if tensor::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        Ok(result
            .with_grad_fn(Some(create_sub_function(lhs, rhs)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Matrix multiplication with automatic differentiation
///
/// This function performs matrix multiplication and automatically attaches
/// a `MatMulFunction` to the result tensor if either input requires gradients.
///
/// # Arguments
/// * `lhs` - Left-hand side tensor
/// * `rhs` - Right-hand side tensor
///
/// # Returns
/// Result tensor with automatic differentiation support
#[allow(clippy::missing_errors_doc)]
pub fn matmul<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType
        + Clone
        + Copy
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
{
    // Convert to dense for operation
    let lhs_dense = lhs
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let rhs_dense = rhs
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    // Perform matrix multiplication on dense tensors
    let result_dense = tensor::ops::linalg::matmul(&lhs_dense, &rhs_dense)
        .map_err(crate::AutogradError::TensorError)?;

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        Ok(result
            .with_grad_fn(Some(create_matmul_function(lhs, rhs)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Mean reduction with automatic differentiation
///
/// This function computes the mean of tensor elements and automatically attaches
/// a `MeanFunction` to the result tensor if the input requires gradients.
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Optional dimensions to reduce. If None, reduces all dimensions.
/// * `keepdim` - Whether to keep reduced dimensions
///
/// # Returns
/// Result tensor with automatic differentiation support
#[allow(clippy::missing_errors_doc)]
pub fn mean<B, S, T>(
    input: &Tensor<B, S, T>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,

    T: DataType
        + Clone
        + Copy
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::One
        + dtype::traits::FloatExt
        + num_traits::FromPrimitive
        + 'static,
{
    // Convert to dense for operation
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    // Perform mean operation on dense tensor
    let result_dense = tensor::ops::reduction::mean(&input_dense, dim, keepdim)
        .map_err(crate::AutogradError::TensorError)?;

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_mean_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Sum reduction with automatic differentiation
///
/// This function computes the sum of tensor elements and automatically attaches
/// a `SumFunction` to the result tensor if the input requires gradients.
///
/// # Arguments
/// * `input` - Input tensor
/// * `dim` - Optional dimensions to reduce. If None, reduces all dimensions.
/// * `keepdim` - Whether to keep reduced dimensions
///
/// # Returns
/// Result tensor with automatic differentiation support
#[allow(clippy::missing_errors_doc)]
pub fn sum<B, S, T>(
    input: &Tensor<B, S, T>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + 'static,
{
    // Convert to dense for operation
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;

    // Perform sum operation on dense tensor
    let result_dense = tensor::ops::reduction::sum(&input_dense, dim, keepdim)
        .map_err(crate::AutogradError::TensorError)?;

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_sum_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise division with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn div<B, S, T>(lhs: &Tensor<B, S, T>, rhs: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T> + Clone + Send + Sync + 'static,
    T: DataType + std::ops::Div<Output = T> + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform the division operation using the arithmetic module directly
    let result =
        tensor::ops::arithmetic::div(lhs, rhs).map_err(crate::AutogradError::TensorError)?;
    if tensor::tensor_core::grad_enabled() && (lhs.requires_grad() || rhs.requires_grad()) {
        Ok(result
            .with_grad_fn(Some(create_div_function(lhs, rhs)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise negation with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn neg<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + std::ops::Neg<Output = T> + Copy + 'static + dtype::traits::FloatExt,
{
    let result = tensor::ops::arithmetic::neg(input).map_err(crate::AutogradError::TensorError)?;
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_neg_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise power with scalar exponent with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn pow<B, S, T>(input: &Tensor<B, S, T>, exponent: f64) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + FloatExt + FromPrimitive,
{
    // Perform power operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let data = input_dense.storage().as_slice();
    let mut res_data = Vec::with_capacity(data.len());

    for &val in data {
        let val_f64 = val
            .to_f64()
            .ok_or_else(|| crate::AutogradError::NumericalError {
                details: "pow: failed to convert input element to f64".to_string(),
            })?;
        let res = val_f64.powf(exponent);
        let res_t = T::from_f64(res).ok_or_else(|| crate::AutogradError::NumericalError {
            details: "pow: failed to convert pow result from f64".to_string(),
        })?;
        res_data.push(res_t);
    }

    let result = Tensor::<B, S, T>::from_vec(res_data, input_dense.shape().dims())
        .map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_pow_function(input, exponent)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

fn create_exp_function<B, S, T>(
    input: &Tensor<B, S, T>,
    output: &Tensor<B, S, T>,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(ExpFunction::new(Arc::new(input.clone()), Arc::new(output.clone())))
}

fn create_log_function<B, S, T>(input: &Tensor<B, S, T>) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(LogFunction::new(Arc::new(input.clone())))
}

/// Element-wise exponential with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn exp<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform exp operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let result_dense = input_dense.exp();

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        let grad_fn = create_exp_function(input, &result);
        Ok(result
            .with_grad_fn(Some(grad_fn))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise natural logarithm with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn log<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform log operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let result_dense = input_dense.log();

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_log_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

fn create_sin_function<B, S, T>(input: &Tensor<B, S, T>) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(SinFunction::new(Arc::new(input.clone())))
}

fn create_cos_function<B, S, T>(input: &Tensor<B, S, T>) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    Arc::new(CosFunction::new(Arc::new(input.clone())))
}

/// Element-wise sine with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn sin<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform sin operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let result_dense = input_dense.sin();

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_sin_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Element-wise cosine with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn cos<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt,
{
    // Perform cos operation via dense fallback
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let result_dense = input_dense.cos();

    // Convert back to original storage type
    let data = result_dense.as_slice().to_vec();
    let dims = result_dense.shape().dims();
    let result =
        Tensor::<B, S, T>::from_vec(data, dims).map_err(crate::AutogradError::TensorError)?;

    // Create computation graph if gradients are required
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_cos_function(input)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

fn create_max_function<B, S, T>(
    input: &Tensor<B, S, T>,
    mask: &Tensor<B, S, T>,
    dim: usize,
    keepdim: bool,
) -> Arc<dyn tensor::Function<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + dtype::traits::FloatExt + Copy + 'static,
{
    Arc::new(MaxFunction::new(
        Arc::new(input.clone()),
        Arc::new(mask.clone()),
        dim,
        keepdim,
    ))
}

/// Maximum reduction along dimension with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn max<B, S, T>(input: &Tensor<B, S, T>, dim: usize, keepdim: bool) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static + dtype::traits::FloatExt + PartialOrd,
{
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let input_shape = input_dense.shape().dims();

    if dim >= input_shape.len() {
        return Err(crate::AutogradError::InvalidInput {
            message: format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                input_shape.len()
            ),
        });
    }

    // Manual reduction implementation

    // Calculate output shape
    let mut output_shape = input_shape.to_vec();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    let output_size: usize = output_shape.iter().product();
    // Use neg_infinity for float types
    let min_val = T::neg_infinity();
    let mut output_data = vec![min_val; output_size];

    let ndim = input_shape.len();

    let input_data_slice = input_dense.storage().as_slice();

    // Iterate over input and compute corresponding output index
    for (i, &val) in input_data_slice.iter().enumerate() {
        // Convert linear input index i to coords
        let mut coords = vec![0; ndim];
        let mut temp = i;
        for d in (0..ndim).rev() {
            coords[d] = temp % input_shape[d];
            temp /= input_shape[d];
        }

        // Convert coords to output linear index
        let mut out_idx = 0;
        let mut current_stride = 1;

        // Output shape matches input shape except at dim.
        // We iterate output dims in reverse to match linear layout
        for d in (0..output_shape.len()).rev() {
            let out_coord = if keepdim {
                if d == dim {
                    0
                } else {
                    coords[d]
                }
            } else {
                let input_d = if d < dim { d } else { d + 1 };
                coords[input_d]
            };

            out_idx += out_coord * current_stride;
            current_stride *= output_shape[d];
        }

        if val > output_data[out_idx] {
            output_data[out_idx] = val;
        }
    }

    // Now create result tensor
    let result = Tensor::<B, S, T>::from_vec(output_data.clone(), &output_shape)
        .map_err(crate::AutogradError::TensorError)?;

    // If gradients required, create mask
    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        let mut mask_data = Vec::with_capacity(input_data_slice.len());
        for (i, &val) in input_data_slice.iter().enumerate() {
            let mut coords = vec![0; ndim];
            let mut temp = i;
            for d in (0..ndim).rev() {
                coords[d] = temp % input_shape[d];
                temp /= input_shape[d];
            }

            let mut out_idx = 0;
            let mut current_stride = 1;
            for d in (0..output_shape.len()).rev() {
                let out_coord = if keepdim {
                    if d == dim {
                        0
                    } else {
                        coords[d]
                    }
                } else {
                    let input_d = if d < dim { d } else { d + 1 };
                    coords[input_d]
                };
                out_idx += out_coord * current_stride;
                current_stride *= output_shape[d];
            }

            if (val - output_data[out_idx]).abs() < T::epsilon() {
                mask_data.push(T::one());
            } else {
                mask_data.push(T::zero());
            }
        }

        let mask = Tensor::<B, S, T>::from_vec(mask_data, input_shape)
            .map_err(crate::AutogradError::TensorError)?;

        Ok(result
            .with_grad_fn(Some(create_max_function(input, &mask, dim, keepdim)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}

/// Reshape tensor with automatic differentiation
#[allow(clippy::missing_errors_doc)]
pub fn reshape<B, S, T>(input: &Tensor<B, S, T>, shape: &[usize]) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + 'static,
    T: DataType + Copy + 'static,
{
    let input_dense = input
        .to_dense_generic()
        .map_err(crate::AutogradError::TensorError)?;
    let input_shape_vec = input.shape().dims().to_vec();

    let current_size: usize = input_shape_vec.iter().product();
    let new_size: usize = shape.iter().product();

    if current_size != new_size {
        return Err(crate::AutogradError::InvalidInput {
            message: format!(
                "Shape mismatch: cannot reshape from {input_shape_vec:?} to {shape:?}"
            ),
        });
    }

    let data = input_dense.storage().as_slice().to_vec();
    let result = Tensor::<B, S, T>::from_vec_with_backend(data, shape, input.backend().clone())
        .map_err(crate::AutogradError::TensorError)?;

    if tensor::tensor_core::grad_enabled() && input.requires_grad() {
        Ok(result
            .with_grad_fn(Some(create_reshape_function(input, input_shape_vec)))
            .requires_grad_(true))
    } else {
        Ok(result)
    }
}
