//! # Coeus Backend
//!
//! Backend abstraction layer providing device-agnostic tensor operations.
//! Supports both CPU and GPU acceleration with automatic dispatch.
//!
//! ## Architecture
//!
//! The backend system is designed around generic traits `B` and `T`:
//! - `B`: Backend type (CpuBackend, GpuBackend, etc.)
//! - `T`: Data type (f32, f64, i32, etc.)
//!
//! This allows for compile-time backend and dtype selection while maintaining
//! runtime flexibility.
//!
//! ## Device Selection
//!
//! ```rust,no_run
//! use coeus_backend::{Backend, CpuBackend, GpuBackend};
//!
//! // CPU backend (default)
//! let cpu_backend = CpuBackend::new();
//!
//! // GPU backend with true hardware acceleration using WGSL compute shaders
//! // Note: GPU backend creation requires async context
//! // let gpu_backend = GpuBackend::new().await.unwrap(); // ✅ Real GPU acceleration
//!
//! // Generic function that works with any backend
//! // async fn tensor_ops<B: Backend<f32> + Sync>(backend: &B) {
//! //     let tensor = backend.zeros(&[3, 3]).await.unwrap();
//! // }
//! ```
//!
//! ## Backend Trait B
//!
//! The `Backend<T>` trait provides:
//! - Device-agnostic tensor creation and operations
//! - Memory management (allocation/deallocation)
//! - Data transfer between host and device
//! - Kernel execution for mathematical operations
//!
//! ## Quantization Support
//!
//! Full support for quantized operations:
//! ```rust,no_run
//! use coeus_backend::{Backend, CpuBackend};
//! use coeus_dtype::QuantizedDtype;
//!
//! // Quantized operations require async context
//! // async fn quantized_ops<B: Backend<i8> + Sync>(backend: &B) {
//! //     let scale = i8::scale(); // 1.0/127.0 for symmetric quantization
//! //     let zero_point = i8::zero_point(); // 0 for symmetric
//! //     let tensor = backend.zeros(&[3, 3]).await.unwrap();
//! // }
//! ```
//!
//! ## Type Safety
//!
//! Backends are generic over data types (`T`) and maintain type safety
//! throughout the tensor computation pipeline.
//!
//! ## Performance
//!
//! - **CPU Backend**: Optimized for multi-core CPUs using Rayon
//! - **GPU Backend**: Hardware acceleration with WGSL compute shaders (f32 focus)
//!   - Element-wise operations: GPU-accelerated for f32 (add, sub, mul, div)
//!   - Matrix multiplication: GPU-accelerated GEMM for f32, CPU fallback for others
//!   - Reduction operations: GPU-accelerated sum_dim, mean_dim for 2D tensors
//!   - Concatenation: Implemented with CPU fallback for all types
//!   - Memory transfers: Optimized host-device data movement
//!   - Security: Type-safe operations with proper bounds checking
//! - **Zero-copy**: Where possible, avoids unnecessary data movement
//!
//! ## References
//!
//! - [Burn.rs Backend Architecture](https://github.com/tracel-ai/burn/tree/main/crates/burn-core/src/backend)
//! - [wgpu GPU Abstraction](https://wgpu.rs/)
//! - [Rayon Parallel Computing](https://docs.rs/rayon/latest/rayon/)

pub mod cpu;
pub mod error;
pub mod gpu;

pub use cpu::CpuBackend;
pub use error::{BackendError, Result};
pub use gpu::GpuBackend;

use coeus_dtype::Dtype;
use std::sync::Arc;

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU device for multi-core processing
    #[default]
    Cpu,
    /// GPU device for parallel computation
    Gpu,
}

/// Core Backend trait B providing device-agnostic tensor operations
///
/// This trait defines the interface for backend implementations (B) that can operate
/// on tensors of type T. It's designed similar to Burn.rs backend trait system.
///
/// # Type Parameters
///
/// * `T` - The data type for tensor elements (must implement `Dtype`)
///
/// # Backend Pattern
///
/// The backend pattern allows for compile-time backend selection:
/// ```rust,no_run
/// use coeus_backend::Backend;
/// use coeus_backend::CpuBackend;
///
/// // Generic function that works with any backend
/// async fn tensor_ops<B: Backend<f32> + Sync>(backend: &B) {
///     let tensor = backend.zeros(&[3, 3]).await.unwrap();
///     println!("Created tensor with shape: {:?}", tensor.shape());
/// }
///
/// // Call with CPU backend
/// let cpu_backend = CpuBackend::new();
/// tokio::spawn(async move {
///     tensor_ops(&cpu_backend).await;
/// });
/// ```
///
/// # Quantization Support
///
/// Backends support quantized operations through the dtype system:
/// ```rust,no_run
/// use coeus_backend::Backend;
/// use coeus_backend::CpuBackend;
/// use coeus_dtype::QuantizedDtype;
///
/// // Quantized operations with i8 tensors
/// async fn quantized_ops<B: Backend<i8> + Sync>(backend: &B) {
///     let tensor = backend.zeros(&[3, 3]).await.unwrap();
///     // i8 tensor with quantization parameters
///     let scale = 1.0 / 127.0; // Symmetric quantization scale
///     let zero_point = 0i8; // Symmetric quantization zero point
///     println!("Quantized tensor scale: {}, zero_point: {}", scale, zero_point);
/// }
///
/// // Call with CPU backend for quantized operations
/// let cpu_backend = CpuBackend::new();
/// tokio::spawn(async move {
///     quantized_ops(&cpu_backend).await;
/// });
/// ```
#[async_trait::async_trait]
pub trait Backend<T: Dtype> {
    /// Device this backend operates on
    fn device(&self) -> Device;

    /// Allocate memory for a tensor with given shape
    async fn allocate(&self, shape: &[usize]) -> Result<Arc<TensorData<T>>>;

    /// Create a tensor filled with zeros
    async fn zeros(&self, shape: &[usize]) -> Result<Tensor<T>> {
        let data = self.allocate(shape).await?;
        Ok(Tensor {
            data,
            shape: shape.to_vec(),
        })
    }

    /// Create a tensor filled with ones
    async fn ones(&self, shape: &[usize]) -> Result<Tensor<T>> {
        let data = self.allocate(shape).await?;
        Ok(Tensor {
            data,
            shape: shape.to_vec(),
        })
    }

    /// Copy data from host to device
    async fn copy_from_host(&self, data: &[T], shape: &[usize]) -> Result<Tensor<T>>;

    /// Copy data from device to host
    async fn copy_to_host(&self, tensor: &Tensor<T>) -> Result<Vec<T>>;

    /// Element-wise addition
    async fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Element-wise subtraction
    async fn sub(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Element-wise multiplication
    async fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Element-wise division
    async fn div(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Matrix multiplication
    async fn matmul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;

    /// Transpose tensor
    async fn transpose(&self, tensor: &Tensor<T>, dim0: usize, dim1: usize) -> Result<Tensor<T>>;

    /// Sum along specified dimensions
    async fn sum_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>>;

    /// Mean along specified dimensions
    async fn mean_dim(&self, tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>>;

    /// Concatenate tensors along specified dimension
    async fn cat(&self, tensors: &[&Tensor<T>], dim: usize) -> Result<Tensor<T>>;
}

/// Extension trait for backend operations with quantization support
///
/// This trait provides additional operations that work with quantized types
/// and support conversion between quantized and floating-point representations.
#[async_trait::async_trait]
pub trait QuantizedBackend<T: Dtype>: Backend<T> {
    /// Quantize floating-point tensor to quantized representation
    async fn quantize<Q: Dtype + coeus_dtype::QuantizedDtype>(
        &self,
        tensor: &Tensor<f32>,
        scale: f32,
        zero_point: Q,
    ) -> Result<Tensor<Q>>;

    /// Dequantize quantized tensor back to floating-point
    async fn dequantize<Q: Dtype + coeus_dtype::QuantizedDtype>(
        &self,
        tensor: &Tensor<Q>,
        scale: f32,
        zero_point: Q,
    ) -> Result<Tensor<f32>>;
}

/// Generic tensor operations that work across all backends
///
/// This module provides high-level tensor operations that automatically
/// dispatch to the appropriate backend implementation.
pub mod ops {
    use super::*;

    /// Generic tensor addition that works with any backend
    pub async fn add<B: Backend<T> + Sync, T: Dtype>(
        backend: &B,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        backend.add(a, b).await
    }

    /// Generic matrix multiplication
    pub async fn matmul<B: Backend<T> + Sync, T: Dtype>(
        backend: &B,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        backend.matmul(a, b).await
    }

    /// Generic tensor creation from data
    pub async fn from_data<B: Backend<T> + Sync, T: Dtype>(
        backend: &B,
        data: &[T],
        shape: &[usize],
    ) -> Result<Tensor<T>> {
        backend.copy_from_host(data, shape).await
    }

    /// Create zeros tensor with any backend
    pub async fn zeros<B: Backend<T> + Sync, T: Dtype>(
        backend: &B,
        shape: &[usize],
    ) -> Result<Tensor<T>> {
        backend.zeros(shape).await
    }
}

/// Tensor data container (device-specific implementation)
pub struct TensorData<T: Dtype> {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Device-specific data storage
    pub data: BackendData<T>,
}

/// Device-specific data storage
pub enum BackendData<T: Dtype> {
    /// CPU memory buffer
    Cpu(Vec<T>),
    /// GPU buffer (wgpu)
    Gpu(wgpu::Buffer),
}

/// Tensor wrapper providing high-level operations
pub struct Tensor<T: Dtype> {
    /// Tensor data (device-specific)
    pub data: Arc<TensorData<T>>,
    /// Cached shape for convenience
    pub shape: Vec<usize>,
}

impl<T: Dtype> Tensor<T> {
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is scalar (shape = [])
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Get scalar value (panics if not scalar)
    pub fn item(&self) -> T {
        assert!(self.is_scalar(), "Tensor must be scalar to call item()");
        // Implementation-specific data access
        match &self.data.data {
            BackendData::Cpu(data) => data[0],
            BackendData::Gpu(_) => panic!("Cannot access GPU tensor data directly"),
        }
    }
}

impl<T: Dtype> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
        }
    }
}

/// Helper macro for backend implementations
#[macro_export]
macro_rules! impl_backend_common {
    ($backend:ty) => {
        #[async_trait::async_trait]
        impl<T: Dtype> Backend<T> for $backend {
            fn device(&self) -> Device {
                // Implementation-specific device type
                Device::Cpu // Default to CPU
            }

            async fn allocate(&self, shape: &[usize]) -> Result<Arc<TensorData<T>>> {
                let numel = shape.iter().product();
                let data = vec![T::zero(); numel]; // Allocate with zeros
                Ok(Arc::new(TensorData {
                    shape: shape.to_vec(),
                    data: BackendData::Cpu(data),
                }))
            }

            async fn copy_from_host(&self, data: &[T], shape: &[usize]) -> Result<Tensor<T>> {
                let tensor_data = Arc::new(TensorData {
                    shape: shape.to_vec(),
                    data: BackendData::Cpu(data.to_vec()),
                });
                Ok(Tensor {
                    data: tensor_data,
                    shape: shape.to_vec(),
                })
            }

            async fn copy_to_host(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
                match &tensor.data.data {
                    BackendData::Cpu(data) => Ok(data.clone()),
                    BackendData::Gpu(_) => Err(BackendError::DeviceMismatch {
                        required: Device::Cpu,
                        actual: Device::Gpu,
                    }),
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(<CpuBackend as Backend<f32>>::device(&backend), Device::Cpu);
    }

    #[tokio::test]
    async fn test_tensor_creation() {
        let backend = CpuBackend::new();
        let tensor: Tensor<f32> = backend.zeros(&[2, 3]).await.unwrap();

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert!(!tensor.is_scalar());
    }

    #[tokio::test]
    async fn test_scalar_tensor() {
        let backend = CpuBackend::new();
        let tensor: Tensor<f32> = backend.zeros(&[]).await.unwrap();

        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.numel(), 1);
        assert!(tensor.is_scalar());
    }
}
