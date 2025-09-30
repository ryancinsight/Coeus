//! Tensor traits and type definitions
//!
//! This module defines the core traits that tensor types should implement,
//! providing a common interface for tensor operations across different backends
//! and storage formats.

use crate::{Dtype, Result, Tensor};
use coeus_backend::Backend;
use coeus_storage::{TensorStorage, DenseStorage};

/// Marker trait for tensors that support autograd operations
pub trait AutogradTensor<T: Dtype, B: Backend<T> + Clone + Send + Sync> {
    /// Perform backward pass from this tensor
    fn backward(&self) -> Result<()>;

    /// Get computational graph node ID
    fn node_id(&self) -> Option<u64>;

    /// Set computational graph node ID
    fn set_node_id(&mut self, node_id: Option<u64>);
}

/// Trait for tensor creation operations
pub trait TensorFrom<T: Dtype, B: Backend<T> + Clone + Send + Sync> {
    /// Create tensor from vector and shape
    fn from_vec(backend: B, data: Vec<T>, shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Create tensor filled with zeros
    fn zeros(backend: B, shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Create tensor filled with ones
    fn ones(backend: B, shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Create scalar tensor
    fn scalar(backend: B, value: T) -> Result<Self>
    where
        Self: Sized;

    /// Create identity matrix
    fn eye(backend: B, size: usize) -> Result<Self>
    where
        Self: Sized;
}

/// Core tensor trait with storage-generic operations
///
/// This trait provides a unified interface for tensor operations that work across
/// different storage formats (dense, sparse) and backends. It enables zero-cost
/// polymorphism for tensor operations while maintaining compile-time type safety.
///
/// # Generic Parameters
/// - `T`: Data type implementing `Dtype` trait
/// - `B`: Backend implementing `Backend<T>` trait
/// - `S`: Storage format implementing `TensorStorage<T>` trait
pub trait TensorTrait<T, B, S>
where
    T: Dtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T> + Clone + Send + Sync,
{
    /// Get immutable reference to storage
    fn storage(&self) -> &S;

    /// Get mutable reference to storage
    fn storage_mut(&mut self) -> &mut S;

    /// Get immutable reference to backend
    fn backend(&self) -> &B;

    /// Get mutable reference to backend
    fn backend_mut(&mut self) -> &mut B;

    /// Get tensor shape
    fn shape<'a>(&'a self) -> &'a [usize] where S: 'a {
        self.storage().shape()
    }

    /// Get number of elements
    fn numel(&self) -> usize {
        self.storage().numel()
    }

    /// Check if tensor is contiguous in memory
    fn is_contiguous(&self) -> bool {
        self.storage().is_contiguous()
    }

    /// Check if tensor is sparse
    fn is_sparse(&self) -> bool {
        self.storage().is_sparse()
    }

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize {
        self.storage().memory_usage()
    }

    /// Validate tensor integrity
    fn validate(&self) -> Result<()> {
        self.storage().validate().map_err(|e| crate::TensorError::StorageError(e.to_string()))
    }

    /// Convert to dense representation (may allocate)
    fn to_dense(&self) -> Vec<T> {
        self.storage().to_dense()
    }

    /// Element-wise addition with another tensor
    ///
    /// This operation works across different storage formats by converting
    /// sparse tensors to dense when necessary for computation.
    fn add(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise multiplication with another tensor
    fn mul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise division
    fn div(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Matrix multiplication (gemm operation)
    ///
    /// For sparse tensors, this may convert to dense for computation
    /// depending on backend capabilities.
    fn matmul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise negation
    fn neg(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise exponential
    fn exp(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise natural logarithm
    fn log(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise sine
    fn sin(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise cosine
    fn cos(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise hyperbolic tangent
    fn tanh(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise sigmoid (1 / (1 + exp(-x)))
    fn sigmoid(&self) -> Result<Self>
    where
        Self: Sized;

    /// Sum reduction along specified dimensions
    fn sum(&self, dims: Option<&[usize]>) -> Result<Self>
    where
        Self: Sized;

    /// Mean reduction along specified dimensions
    fn mean(&self, dims: Option<&[usize]>) -> Result<Self>
    where
        Self: Sized;

    /// Maximum reduction along specified dimensions
    fn max(&self, dims: Option<&[usize]>) -> Result<Self>
    where
        Self: Sized;

    /// Minimum reduction along specified dimensions
    fn min(&self, dims: Option<&[usize]>) -> Result<Self>
    where
        Self: Sized;

    /// Transpose tensor along specified dimensions
    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self>
    where
        Self: Sized;

    /// Reshape tensor to new shape
    fn reshape(&self, new_shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Squeeze singleton dimensions
    fn squeeze(&self) -> Result<Self>
    where
        Self: Sized;

    /// Add singleton dimensions at specified positions
    fn unsqueeze(&self, dim: usize) -> Result<Self>
    where
        Self: Sized;

    /// Create a copy of the tensor
    fn clone_tensor(&self) -> Self
    where
        Self: Sized;
}

impl<T, B, S> TensorTrait<T, B, S> for crate::Tensor<T, B, S>
where
    T: Dtype + std::ops::Neg<Output = T> + num_traits::Float + std::iter::Sum<T> + num_traits::FromPrimitive + Clone + coeus_dtype::FloatDtype,
    B: Backend<T> + Clone + Send + Sync,
    S: TensorStorage<T> + Clone + Send + Sync,
{
    fn storage(&self) -> &S {
        &self.storage
    }

    fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    fn backend(&self) -> &B {
        &self.backend
    }

    fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    fn add(&self, other: &Self) -> Result<Self>
    {
        // For now, require dense computation for addition
        if self.is_sparse() || other.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Addition with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::add(self, other)
    }

    fn mul(&self, other: &Self) -> Result<Self>
    {
        // For now, require dense computation for multiplication
        if self.is_sparse() || other.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Multiplication with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::mul(self, other)
    }

    fn sub(&self, other: &Self) -> Result<Self>
    {
        // For now, require dense computation for subtraction
        if self.is_sparse() || other.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Subtraction with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::sub(self, other)
    }

    fn div(&self, other: &Self) -> Result<Self> {
        // For now, require dense computation for division
        if self.is_sparse() || other.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Division with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::div(self, other)
    }

    fn matmul(&self, other: &Self) -> Result<Self>
    {
        // For now, require dense computation for matrix multiplication
        if self.is_sparse() || other.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Matrix multiplication with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::matrix::matmul_impl(self, other)
    }

    fn neg(&self) -> Result<Self>
    {
        // For now, require dense computation for negation
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Negation with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::neg(self)
    }

    fn exp(&self) -> Result<Self>
    {
        // For now, require dense computation for exponential
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Exponential with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::exp(self)
    }

    fn log(&self) -> Result<Self>
    {
        // For now, require dense computation for logarithm
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Log operation on sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::log(self)
    }

    fn sin(&self) -> Result<Self>
    {
        // For now, require dense computation for sine
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Sine operation on sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::sin(self)
    }

    fn cos(&self) -> Result<Self>
    {
        // For now, require dense computation for cosine
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Cosine operation on sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::arithmetic::cos(self)
    }

    fn tanh(&self) -> Result<Self>
    {
        // For now, require dense computation for tanh
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Tanh operation on sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::activations::tanh(self)
    }

    fn sigmoid(&self) -> Result<Self>
    {
        // For now, require dense computation for sigmoid
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Sigmoid operation on sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::activations::sigmoid(self)
    }

    fn sum(&self, _dims: Option<&[usize]>) -> Result<Self>
    {
        // For now, require dense computation for sum reduction
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Sum reduction with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::reduction::sum(self)
    }

    fn mean(&self, _dims: Option<&[usize]>) -> Result<Self>
    {
        // For now, require dense computation for mean reduction
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Mean reduction with sparse tensors requires dense conversion".to_string()
            ));
        }
        crate::ops::reduction::mean(self)
    }

    fn max(&self, _dims: Option<&[usize]>) -> Result<Self>
    {
        // Max reduction not yet implemented
        Err(crate::TensorError::NotImplemented("Max reduction not yet implemented".to_string()))
    }

    fn min(&self, _dims: Option<&[usize]>) -> Result<Self>
    {
        // Min reduction not yet implemented
        Err(crate::TensorError::NotImplemented("Min reduction not yet implemented".to_string()))
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        // For now, require dense computation for transpose
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Transpose with sparse tensors requires dense conversion".to_string()
            ));
        }
        Tensor::transpose(self, dim0, dim1)
    }

    fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        // For now, require dense computation for reshape
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Reshape with sparse tensors requires dense conversion".to_string()
            ));
        }
        Tensor::reshape(self, new_shape.to_vec())
    }

    fn squeeze(&self) -> Result<Self> {
        // For now, require dense computation for squeeze
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Squeeze with sparse tensors requires dense conversion".to_string()
            ));
        }
        Tensor::squeeze(self)
    }

    fn unsqueeze(&self, dim: usize) -> Result<Self> {
        // For now, require dense computation for unsqueeze
        if self.is_sparse() {
            return Err(crate::TensorError::SparseOperationNotSupported(
                "Unsqueeze with sparse tensors requires dense conversion".to_string()
            ));
        }
        Tensor::unsqueeze(self, dim)
    }

    fn clone_tensor(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_tensor_trait_implementation() {
        let backend = CpuBackend::new();
        let tensor = crate::Tensor::<f32, CpuBackend>::from_vec(
            backend,
            vec![1.0, 2.0, 3.0],
            vec![3]
        ).unwrap();

        // Test trait methods
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.memory_usage(), 3 * std::mem::size_of::<f32>() + std::mem::size_of::<usize>() * tensor.shape().len());
    }
}