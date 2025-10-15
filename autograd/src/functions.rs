//! PyTorch-compatible Function trait for automatic differentiation
//!
//! This module implements the core Function trait that enables automatic graph
//! construction and gradient computation compatible with PyTorch's autograd system.
//!
//! ## Zero-Cost Generics Support
//!
//! All Function implementations support the full B<S<T>> generic hierarchy:
//! - **B**: Backend (CpuBackend, GpuBackend, etc.)
//! - **S**: Storage (DenseStorage, SparseStorage, etc.)
//! - **T**: DataType (Float32, Float64, etc.)
//!
//! This enables compile-time specialization for optimal performance across
//! different hardware and data configurations.

use crate::error::{AutogradError, Result};
extern crate alloc;
use alloc::{sync::Arc, vec::Vec};
use core::fmt::Debug;
use std::ops::{Add, Div, Mul};
use coeus_dtype::float::Float32;
use coeus_dtype::traits::FloatExt;
use coeus_backend::Backend;
use coeus_storage::{Storage, StorageFromVec, StorageToDense};
use coeus_dtype::DataType;
use coeus_tensor::Function;


/// Lightweight tensor reference for Function inputs
///
/// Functions store Arc references to input tensors to access their values
/// during backward pass without storing full tensor data.
///
/// # Generic Support
/// Supports any B<S<T>> combination through trait bounds.
/// Type alias for tensor references used in automatic differentiation
/// Generic over Backend<B>, Storage<S>, and DataType<T>
pub type TensorRef<B, S, T> = Arc<coeus_tensor::Tensor<B, S, T>>;

/// Type alias for the default tensor type used in functions
/// TODO: Make this generic to support full B<S<T>> hierarchy
pub type DefaultTensor = coeus_tensor::Tensor<coeus_backend::CpuBackend, coeus_storage::DenseStorage<Float32>, Float32>;

/// Helper macro for common Function trait implementations
///
/// Reduces boilerplate by implementing the standard traits for Function structs.
macro_rules! impl_function_traits {
    ($name:ident, $backward_name:expr) => {
        impl<B, S, T> DifferentiableFunction<B, S, T> for $name<B, S, T>
        where
            B: Backend + core::fmt::Debug + Send + Sync + 'static,
            S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
            T: DataType,
        {
            fn name(&self) -> &'static str {
                $backward_name
            }
        }

        impl<B, S, T> crate::traits::AsAny for $name<B, S, T>
        where
            B: Backend + core::fmt::Debug + Send + Sync + 'static,
            S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
            T: DataType,
        {
            fn as_any(&self) -> &dyn core::any::Any {
                self
            }
        }

        impl<B, S, T> coeus_tensor::AsAny for $name<B, S, T>
        where
            B: Backend + core::fmt::Debug + Send + Sync + 'static,
            S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
            T: DataType,
        {
            fn as_any(&self) -> &dyn core::any::Any {
                self
            }
        }

        impl<B, S, T> coeus_tensor::DifferentiableFunction<B, S, T> for $name<B, S, T>
        where
            B: Backend + core::fmt::Debug + Send + Sync + 'static,
            S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
            T: DataType,
        {
            fn name(&self) -> &'static str {
                $backward_name
            }
        }

        impl<B, S, T> coeus_tensor::Function<B, S, T> for $name<B, S, T>
        where
            B: Backend + core::fmt::Debug + Send + Sync + 'static,
            S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
            T: DataType,
        {
            fn inputs(&self) -> &[TensorRef<B, S, T>] {
                &self.inputs
            }

            fn backward(&self, _grad_output: &coeus_tensor::Tensor<B, S, T>) -> anyhow::Result<Vec<coeus_tensor::Tensor<B, S, T>>> {
                // Generic backward implementation not yet available
                // Return error for unsupported generic operations
                anyhow::bail!(crate::error::AutogradError::NotImplemented {
                    operation: concat!(stringify!($name), " backward for generic tensors").to_string(),
                })
            }
        }

    };
}

/// Marker trait for differentiable functions that can be stored in tensors
///
/// This trait is implemented by Function types and allows tensors to reference
/// their creator functions for automatic differentiation. Extends the tensor crate's
/// DifferentiableFunction with additional functionality.
pub trait DifferentiableFunction<B, S, T>: coeus_tensor::DifferentiableFunction<B, S, T> + crate::traits::AsAny
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Get the name of this function for debugging
    fn name(&self) -> &'static str;
}

/// Type-erased function reference for tensor grad_fn fields
/// Uses the existing DifferentiableFunction trait for compatibility
pub type FunctionRef<B, S, T> = Arc<dyn DifferentiableFunction<B, S, T>>;

/// PyTorch-compatible Function trait for automatic differentiation
///
/// Each differentiable operation implements this trait to enable:
/// - Automatic graph construction during forward pass
/// - Gradient computation via backward() method
/// - Memory-efficient representation (stores lightweight input references)
///

/// Base Function implementation for element-wise addition
#[derive(Debug)]
pub struct AddFunction<B, S, T>
where
    B: Backend + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> AddFunction<B, S, T>
where
    B: Backend + core::fmt::Debug + Send + Sync + 'static,
    S: Storage<T> + core::fmt::Debug + Send + Sync + 'static,
    T: DataType,
{
    /// Create a new Add function with input references
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side input tensor
    /// * `rhs` - Right-hand side input tensor
    #[must_use]
    pub fn new(lhs: TensorRef<B, S, T>, rhs: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl_function_traits!(AddFunction, "AddBackward");

/// Base Function implementation for element-wise multiplication
#[derive(Debug)]
pub struct MulFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> MulFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Create a new Mul function with input references
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side input tensor
    /// * `rhs` - Right-hand side input tensor
    #[must_use]
    pub fn new(lhs: TensorRef<B, S, T>, rhs: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }

    pub fn backward(&self, grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>>
    where
        B: Backend + Default + Clone,
        S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
        T: DataType + Add<Output = T> + Mul<Output = T> + Clone,
    {
        // For f = a * b, ∂f/∂a = b, ∂f/∂b = a
        let lhs = &self.inputs[0];
        let rhs = &self.inputs[1];

        // For multiplication, we need element-wise operations
        // grad_lhs = grad_output * rhs, grad_rhs = grad_output * lhs
        // For now, implement simple case where tensors have same shape
        if lhs.shape() == rhs.shape() && lhs.shape() == grad_output.shape() {
            // Convert all inputs to dense for element-wise operations
            let lhs_dense = lhs.to_dense_generic()
                .map_err(|e| AutogradError::TensorError(e))?;
            let rhs_dense = rhs.to_dense_generic()
                .map_err(|e| AutogradError::TensorError(e))?;
            let grad_output_dense = grad_output.to_dense_generic()
                .map_err(|e| AutogradError::TensorError(e))?;

            // Element-wise multiplication on dense data
            let mut grad_lhs_data = Vec::with_capacity(lhs_dense.len());
            let mut grad_rhs_data = Vec::with_capacity(rhs_dense.len());

            let lhs_data = lhs_dense.as_slice();
            let rhs_data = rhs_dense.as_slice();
            let grad_data = grad_output_dense.as_slice();

            for i in 0..lhs_data.len() {
                grad_lhs_data.push(grad_data[i].clone() * rhs_data[i].clone());
                grad_rhs_data.push(grad_data[i].clone() * lhs_data[i].clone());
            }

            // Create gradient tensors with the same storage type as inputs
            let grad_lhs_storage = S::from_vec(grad_lhs_data, lhs.shape().dims())
                .map_err(|e| AutogradError::TensorError(
                    coeus_tensor::TensorError::StorageError(e)
                ))?;
            let grad_rhs_storage = S::from_vec(grad_rhs_data, rhs.shape().dims())
                .map_err(|e| AutogradError::TensorError(
                    coeus_tensor::TensorError::StorageError(e)
                ))?;

            let grad_lhs = coeus_tensor::Tensor::from_storage(
                grad_lhs_storage,
                lhs.backend().clone()
            );
            let grad_rhs = coeus_tensor::Tensor::from_storage(
                grad_rhs_storage,
                rhs.backend().clone()
            );

            Ok(vec![grad_lhs, grad_rhs])
        } else {
            // Broadcasting case - not implemented yet
            Err(AutogradError::NotImplemented {
                operation: "MulFunction backward with broadcasting".to_string(),
            })
        }
    }
}

impl_function_traits!(MulFunction, "MulBackward");

/// Base Function implementation for matrix multiplication
#[derive(Debug)]
pub struct MatMulFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> MatMulFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Create a new MatMul function with input references
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side input tensor (A in A @ B)
    /// * `rhs` - Right-hand side input tensor (B in A @ B)
    #[must_use]
    pub fn new(lhs: TensorRef<B, S, T>, rhs: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }

    pub fn backward(&self, _grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>> {
        // Matrix multiplication gradients require transpose and matmul operations
        // Generic implementation requires complex tensor operations
        Err(crate::error::AutogradError::NotImplemented {
            operation: "MatMulFunction backward for generic tensors".to_string(),
        })
    }
}

impl_function_traits!(MatMulFunction, "MatMulBackward");

/// Base Function implementation for sum reduction
#[derive(Debug)]
pub struct SumFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> SumFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Create a new Sum function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor being summed
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>>
    where
        B: Backend + Clone + Default,
        S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
        T: DataType + Clone,
    {
        // Sum reduction gradient: ∂sum(x)/∂x = 1
        // The gradient w.r.t. input is grad_output broadcasted to input shape
        let input = &self.inputs[0];

        // For sum reduction, gradient is grad_output broadcasted to input shape
        // If grad_output is scalar (typical case), broadcast to input shape
        let grad_input_data: Vec<T> = if grad_output.len() == 1 {
            // Scalar gradient - broadcast to input shape
            vec![grad_output.as_slice()[0].clone(); input.len()]
        } else {
            // Non-scalar gradient - this would require more complex broadcasting
            return Err(AutogradError::NotImplemented {
                operation: "SumFunction backward with non-scalar grad_output".to_string(),
            });
        };

        let grad_input_storage = S::from_vec(grad_input_data, input.shape().dims())
            .map_err(|e| AutogradError::TensorError(
                coeus_tensor::TensorError::StorageError(e)
            ))?;

        let grad_input = coeus_tensor::Tensor::from_storage(
            grad_input_storage,
            input.backend().clone()
        );

        Ok(vec![grad_input])
    }
}

impl_function_traits!(SumFunction, "SumBackward");

/// Base Function implementation for mean reduction
#[derive(Debug)]
pub struct MeanFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> MeanFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Create a new Mean function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor being averaged
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>>
    where
        B: Backend + Clone + Default,
        S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
        T: DataType + Div<Output = T> + Clone,
    {
        // Mean gradient: ∂mean(x)/∂xᵢ = 1/n for all i, where n is total elements
        // The gradient is grad_output / n broadcasted to input shape
        let input = &self.inputs[0];
        let n = input.len() as f64; // Number of elements

        // For now, assume grad_output is scalar (typical case)
        if grad_output.len() != 1 {
            return Err(AutogradError::NotImplemented {
                operation: "MeanFunction backward with non-scalar grad_output".to_string(),
            });
        }

        // Compute grad_output / n
        // For mean gradient, divide grad_output by number of elements
        let grad_scalar = grad_output.as_slice()[0].clone();
        let n_scalar = T::from(n).ok_or_else(|| AutogradError::InvalidInput {
            message: "Cannot convert element count to data type".to_string(),
        })?;

        // grad_input_value = grad_scalar / n_scalar
        let grad_input_value = grad_scalar / n_scalar;

        // Broadcast to input shape
        let grad_input_data: Vec<T> = vec![grad_input_value; input.len()];

        let grad_input_storage = S::from_vec(grad_input_data, input.shape().dims())
            .map_err(|e| AutogradError::TensorError(
                coeus_tensor::TensorError::StorageError(e)
            ))?;

        let grad_input = coeus_tensor::Tensor::from_storage(
            grad_input_storage,
            input.backend().clone()
        );

        Ok(vec![grad_input])
    }
}

impl_function_traits!(MeanFunction, "MeanBackward");


/// Base Function implementation for element-wise exponential
#[derive(Debug)]
pub struct ExpFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> ExpFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Create a new Exp function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor to exponentiate
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>>
    where
        B: Backend + Clone + Default,
        S: Storage<T> + StorageFromVec<T> + Clone + StorageToDense<T> + 'static,
        T: DataType + FloatExt + Mul<Output = T> + Clone,
    {
        // Exponential gradient: ∂exp(x)/∂x = exp(x)
        // The gradient is grad_output * exp(input)
        let input = &self.inputs[0];

        // Compute exp(input) element-wise
        // Convert to dense for computation, then back to original storage type
        let input_dense = input.to_dense_generic()
            .map_err(|e| AutogradError::TensorError(e))?;
        let grad_output_dense = grad_output.to_dense_generic()
            .map_err(|e| AutogradError::TensorError(e))?;

        let mut exp_data = Vec::with_capacity(input_dense.len());
        for &val in input_dense.as_slice() {
            exp_data.push(val.exp());
        }

        // Element-wise multiply with grad_output
        // Assume same shape for now (broadcasting not implemented)
        if input.shape() != grad_output.shape() {
            return Err(AutogradError::InvalidInput {
                message: "ExpFunction backward requires grad_output to have same shape as input".to_string(),
            });
        }

        let mut grad_input_data = Vec::with_capacity(input_dense.len());
        let grad_data = grad_output_dense.as_slice();
        for i in 0..input_dense.len() {
            grad_input_data.push(grad_data[i].clone() * exp_data[i].clone());
        }

        let grad_input_storage = S::from_vec(grad_input_data, input.shape().dims())
            .map_err(|e| AutogradError::TensorError(
                coeus_tensor::TensorError::StorageError(e)
            ))?;

        let grad_input = coeus_tensor::Tensor::from_storage(
            grad_input_storage,
            input.backend().clone()
        );

        Ok(vec![grad_input])
    }
}

impl_function_traits!(ExpFunction, "ExpBackward");

/// Base Function implementation for element-wise natural logarithm
#[derive(Debug)]
pub struct LogFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> LogFunction<B, S, T>
where
    B: Backend + Default,
    S: Storage<T>,
    T: DataType + std::ops::Neg<Output = T> + FloatExt,
{
    /// Create a new Log function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor to take logarithm of
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, _grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>> {
        // Logarithm gradient: ∂log(x)/∂x = 1/x
        // Generic implementation requires powf and division operations
        Err(crate::error::AutogradError::NotImplemented {
            operation: "LogFunction backward for generic tensors".to_string(),
        })
    }
}

impl_function_traits!(LogFunction, "LogBackward");

/// Base Function implementation for element-wise sine
#[derive(Debug)]
pub struct SinFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> SinFunction<B, S, T>
where
    B: Backend + Default,
    S: Storage<T>,
    T: DataType + FloatExt,
{
    /// Create a new Sin function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor to take sine of
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, _grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>> {
        // Sine gradient: ∂sin(x)/∂x = cos(x)
        // Generic implementation requires cos and multiplication operations
        Err(crate::error::AutogradError::NotImplemented {
            operation: "SinFunction backward for generic tensors".to_string(),
        })
    }
}

impl_function_traits!(SinFunction, "SinBackward");

/// Base Function implementation for element-wise cosine
#[derive(Debug)]
pub struct CosFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// References to input tensors for gradient computation
    inputs: Vec<TensorRef<B, S, T>>,
}

impl<B, S, T> CosFunction<B, S, T>
where
    B: Backend + Default,
    S: Storage<T>,
    T: DataType + FloatExt,
{
    /// Create a new Cos function with input reference
    ///
    /// # Arguments
    /// * `input` - Input tensor to take cosine of
    #[must_use]
    pub fn new(input: TensorRef<B, S, T>) -> Self {
        Self {
            inputs: vec![input],
        }
    }

    pub fn backward(&self, _grad_output: &coeus_tensor::Tensor<B, S, T>) -> Result<Vec<coeus_tensor::Tensor<B, S, T>>> {
        // Cosine gradient: ∂cos(x)/∂x = -sin(x)
        // Generic implementation requires sin, negation, and multiplication operations
        Err(crate::error::AutogradError::NotImplemented {
            operation: "CosFunction backward for generic tensors".to_string(),
        })
    }
}

impl_function_traits!(CosFunction, "CosBackward");

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use coeus_tensor::Tensor;
    use coeus_storage::DenseStorage;
    use coeus_dtype::float::Float32;

    fn create_test_tensor(shape: &[usize]) -> TensorRef<coeus_backend::CpuBackend, DenseStorage<Float32>, Float32> {
        let data = vec![Float32::new(1.0); shape.iter().product()];
        let tensor = Tensor::<coeus_backend::CpuBackend, DenseStorage<Float32>, Float32>::from_vec(data, shape).unwrap();
        Arc::new(tensor)
    }

    #[test]
    fn test_add_function_name() {
        let lhs = create_test_tensor(&[2]);
        let rhs = create_test_tensor(&[2]);
        let add_fn = AddFunction::new(lhs, rhs);
        assert_eq!(add_fn.name(), "AddBackward");
    }

    #[test]
    fn test_mul_function_name() {
        let lhs = create_test_tensor(&[2]);
        let rhs = create_test_tensor(&[2]);
        let mul_fn = MulFunction::new(lhs, rhs);
        assert_eq!(mul_fn.name(), "MulBackward");
    }

    #[test]
    fn test_matmul_function_name() {
        let lhs = create_test_tensor(&[2, 3]);
        let rhs = create_test_tensor(&[3, 2]);
        let matmul_fn = MatMulFunction::new(lhs, rhs);
        assert_eq!(matmul_fn.name(), "MatMulBackward");
    }

    #[test]
    fn test_sum_function_name() {
        let input = create_test_tensor(&[2, 3]);
        let sum_fn = SumFunction::new(input);
        assert_eq!(sum_fn.name(), "SumBackward");
    }

    #[test]
    fn test_mean_function_name() {
        let input = create_test_tensor(&[2, 3]);
        let mean_fn = MeanFunction::new(input);
        assert_eq!(mean_fn.name(), "MeanBackward");
    }
}



