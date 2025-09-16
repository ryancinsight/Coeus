//! Backend-integrated tensor implementation
//!
//! This module provides a tensor abstraction that works with the backend system,
//! allowing for device-agnostic tensor operations while maintaining PyTorch-like API.
//!
//! ## Architecture
//!
//! The backend tensor system separates the logical tensor interface from the
//! physical backend implementation:
//!
//! - `BackendTensor<T, B>`: Generic tensor with backend B for type T
//! - Backend trait: Device-agnostic operations
//! - Device selection: Compile-time and runtime backend selection
//!
//! ## Usage
//!
//! ```rust
//! use coeus_tensor::BackendTensor;
//! use coeus_backend::{CpuBackend, Backend};
//!
//! // Create CPU tensor
//! let cpu_tensor = BackendTensor::<f32, CpuBackend>::from_data(
//!     &[1.0, 2.0, 3.0],
//!     &[3]
//! ).await.unwrap();
//!
//! // Perform operations
//! let result = cpu_tensor.add(&cpu_tensor).await.unwrap();
//! ```
//!
//! ## Backend Selection
//!
//! Tensors are generic over their backend, allowing compile-time backend selection:
//!
//! ```rust
//! // CPU tensor
//! type CpuTensor<T> = BackendTensor<T, CpuBackend>;
//!
//! // GPU tensor (when available)
//! type GpuTensor<T> = BackendTensor<T, GpuBackend>;
//!
//! // Function that works with any backend
//! async fn process<B: Backend<f32>>(tensor: &BackendTensor<f32, B>) {
//!     let result = tensor.mul(tensor).await.unwrap();
//! }
//! ```

use crate::{Device as TensorDevice, Layout, Result};
use coeus_backend::{Backend, Device as BackendDevice, Tensor as BackendTensorImpl};
use coeus_dtype::{Dtype, FloatDtype};
use coeus_autograd::NodeId;
use std::sync::Arc;

/// Backend-integrated tensor with PyTorch-like API
///
/// This tensor implementation uses the backend system for device-agnostic
/// operations while providing a familiar PyTorch-style interface.
#[derive(Clone)]
pub struct BackendTensor<T: Dtype + FloatDtype, B: Backend<T>> {
    /// Backend tensor data
    data: BackendTensorImpl<T>,
    /// Backend instance for operations
    backend: B,
    /// Device information (cached for convenience)
    device: TensorDevice,
    /// Layout information
    layout: Layout,
    /// Autograd context for gradient computation
    context: Option<Arc<coeus_autograd::Context<T>>>,
    /// Node ID in the computational graph
    node_id: Option<NodeId>,
    /// Gradient tensor (computed during backward pass)
    grad: Option<Box<Self>>,
}

impl<T: Dtype + FloatDtype, B: Backend<T> + Clone + Sync> BackendTensor<T, B> {
    /// Convert backend Device to tensor Device
    fn convert_device(backend_device: BackendDevice) -> TensorDevice {
        match backend_device {
            BackendDevice::Cpu => TensorDevice::Cpu,
            BackendDevice::Gpu => TensorDevice::Cpu, // Fallback to CPU for now
        }
    }
    /// Create a tensor from data with the given backend
    pub async fn from_data(backend: B, data: &[T], shape: &[usize]) -> Result<Self> {
        let device = Self::convert_device(backend.device());
        let tensor_data = backend.copy_from_host(data, shape).await?;
        Ok(Self {
            data: tensor_data,
            backend,
            device,
            layout: Layout::default(),
            context: None,
            node_id: None,
            grad: None,
        })
    }

    /// Create a tensor filled with zeros
    pub async fn zeros(backend: B, shape: &[usize]) -> Result<Self> {
        let device = Self::convert_device(backend.device());
        let tensor_data = backend.zeros(shape).await?;
        Ok(Self {
            data: tensor_data,
            backend,
            device,
            layout: Layout::default(),
            context: None,
            node_id: None,
            grad: None,
        })
    }

    /// Create a tensor filled with ones
    pub async fn ones(backend: B, shape: &[usize]) -> Result<Self> {
        let device = Self::convert_device(backend.device());
        let tensor_data = backend.ones(shape).await?;
        Ok(Self {
            data: tensor_data,
            backend,
            device,
            layout: Layout::default(),
            context: None,
            node_id: None,
            grad: None,
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Check if tensor is scalar
    pub fn is_scalar(&self) -> bool {
        self.data.is_scalar()
    }

    /// Get scalar value (panics if not scalar)
    pub fn item(&self) -> T {
        self.data.item()
    }

    /// Get device
    pub fn device(&self) -> TensorDevice {
        self.device
    }

    /// Get layout
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Check if this tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.context.is_some()
    }

    /// Enable gradient computation for this tensor
    pub fn requires_grad_mut(&mut self, requires_grad: bool) {
        if requires_grad && self.context.is_none() {
            let context = coeus_autograd::Context::new(true);
            // Create a node for this tensor in the computational graph
            // For now, we'll use a simple identity operation node
            self.node_id = Some(context.next_node_id());
            self.context = Some(Arc::new(context));
        } else if !requires_grad {
            self.context = None;
            self.node_id = None;
        }
    }

    /// Get the gradient tensor
    pub fn grad(&self) -> Option<&Self> {
        self.grad.as_ref().map(|boxed| boxed.as_ref())
    }

    /// Set the gradient tensor
    pub fn set_grad(&mut self, grad: Option<Self>) {
        self.grad = grad.map(Box::new);
    }

    /// Perform backward pass to compute gradients
    pub async fn backward(&mut self) -> Result<()> {
        if let Some(context) = &self.context {
            if let Some(node_id) = self.node_id {
                // Get the initial gradient (defaults to ones if not set)
                let initial_grad = if let Some(existing_grad) = &self.grad {
                    existing_grad.data().to_vec()
                } else {
                    // Default to ones tensor for backward pass
                    self.backend.ones(self.shape()).await?.to_vec()
                };

                // Perform backward computation using the computational graph
                {
                    let mut graph = context.graph().write().await;
                    graph.backward(node_id.0 as u64, initial_grad);
                }

                // Retrieve computed gradients and store them
                let graph = context.graph().read().await;
                if let Some(computed_grad) = graph.get_gradient(node_id.0 as u64) {
                    let grad_data = self.backend.from_vec(computed_grad.clone()).await?;
                    let grad = Self {
                        data: grad_data,
                        backend: self.backend.clone(),
                        device: self.device,
                        layout: self.layout,
                        context: None, // Gradients don't require gradients by default
                        node_id: None,
                        grad: None,
                    };
                    self.grad = Some(Box::new(grad));
                }
            }
        }
        Ok(())
    }

    /// Element-wise addition
    pub async fn add(&self, other: &Self) -> Result<Self> {
        let result_data = self.backend.add(&self.data, &other.data).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Element-wise subtraction
    pub async fn sub(&self, other: &Self) -> Result<Self> {
        let result_data = self.backend.sub(&self.data, &other.data).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Element-wise multiplication
    pub async fn mul(&self, other: &Self) -> Result<Self> {
        let result_data = self.backend.mul(&self.data, &other.data).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Element-wise division
    pub async fn div(&self, other: &Self) -> Result<Self> {
        let result_data = self.backend.div(&self.data, &other.data).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Matrix multiplication
    pub async fn matmul(&self, other: &Self) -> Result<Self> {
        let result_data = self.backend.matmul(&self.data, &other.data).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Transpose tensor
    pub async fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let result_data = self.backend.transpose(&self.data, dim0, dim1).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Sum along specified dimension
    pub async fn sum_dim(&self, dim: usize) -> Result<Self> {
        let result_data = self.backend.sum_dim(&self.data, dim).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Mean along specified dimension
    pub async fn mean_dim(&self, dim: usize) -> Result<Self> {
        let result_data = self.backend.mean_dim(&self.data, dim).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Concatenate tensors along specified dimension
    pub async fn cat(&self, tensors: &[&Self], dim: usize) -> Result<Self> {
        // Convert BackendTensor references to BackendTensorImpl references
        let backend_tensors: Vec<&BackendTensorImpl<T>> = tensors
            .iter()
            .map(|t| &t.data)
            .collect();

        let result_data = self.backend.cat(&backend_tensors, dim).await?;
        Ok(Self {
            data: result_data,
            backend: self.backend.clone(),
            device: Self::convert_device(self.backend.device()),
            layout: self.layout,
            context: self.context.clone(),
            node_id: None, // Operations create new nodes
            grad: None,
        })
    }

    /// Copy tensor data to host (CPU) memory
    pub async fn to_host(&self) -> Result<Vec<T>> {
        Ok(self.backend.copy_to_host(&self.data).await?)
    }

    /// Clone the backend for creating new tensors
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

/// Convenience type aliases for common backend combinations
pub type CpuTensor<T> = BackendTensor<T, coeus_backend::CpuBackend>;
pub type GpuTensor<T> = BackendTensor<T, coeus_backend::GpuBackend>;

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::CpuBackend;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_cpu_tensor_creation() {
        let backend = CpuBackend::new();
        let tensor = BackendTensor::from_data(backend, &[1.0, 2.0, 3.0], &[3]).await.unwrap();

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.numel(), 3);
        assert!(!tensor.is_scalar());
    }

    #[tokio::test]
    async fn test_cpu_tensor_operations() {
        let backend = CpuBackend::new();

        let a = BackendTensor::from_data(backend.clone(), &[1.0, 2.0, 3.0], &[3]).await.unwrap();
        let b = BackendTensor::from_data(backend.clone(), &[4.0, 5.0, 6.0], &[3]).await.unwrap();

        let result = a.add(&b).await.unwrap();
        let data = result.to_host().await.unwrap();

        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[tokio::test]
    async fn test_cpu_tensor_zeros_ones() {
        let backend = CpuBackend::new();

        let zeros = BackendTensor::zeros(backend.clone(), &[2, 3]).await.unwrap();
        let ones = BackendTensor::ones(backend.clone(), &[2, 3]).await.unwrap();

        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);

        // Test that zeros are actually zero (first element)
        let zeros_data: Vec<f32> = zeros.to_host().await.unwrap();
        assert_eq!(zeros_data[0], 0.0);

        // Test that ones are actually one (first element)
        let ones_data: Vec<f32> = ones.to_host().await.unwrap();
        assert_eq!(ones_data[0], 1.0);
    }

    #[tokio::test]
    async fn test_cpu_matrix_multiplication() {
        let backend = CpuBackend::new();

        // 2x3 matrix
        let a = BackendTensor::from_data(
            backend.clone(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3]
        ).await.unwrap();

        // 3x2 matrix
        let b = BackendTensor::from_data(
            backend.clone(),
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 2]
        ).await.unwrap();

        let result = a.matmul(&b).await?.unwrap();
        let data = result.to_host().await.unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert_eq!(data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[tokio::test]
    async fn test_cpu_scalar_tensor() {
        let backend = CpuBackend::new();
        let scalar = BackendTensor::from_data(backend, &[42.0], &[]).await.unwrap();

        assert_eq!(scalar.shape(), &[]);
        assert_eq!(scalar.numel(), 1);
        assert!(scalar.is_scalar());
        assert_eq!(scalar.item(), 42.0);
    }

    #[tokio::test]
    async fn test_cpu_autograd_basic() {
        let backend = CpuBackend::new();
        let mut tensor = BackendTensor::from_data(backend, &[2.0], &[1]).await.unwrap();

        // Initially no gradients required
        assert!(!tensor.requires_grad());
        assert!(tensor.grad().is_none());

        // Enable gradient computation
        tensor.requires_grad_mut(true);
        assert!(tensor.requires_grad());

        // Perform backward pass
        tensor.backward().await.unwrap();

        // Check that gradient was computed
        assert!(tensor.grad().is_some());
        let grad = tensor.grad().unwrap();
        assert_eq!(grad.shape(), &[1]);
        let grad_data: Vec<f32> = grad.to_host().await.unwrap();
        assert_eq!(grad_data[0], 1.0); // Gradient should be 1.0 (d/dx(x) = 1)
    }
}
