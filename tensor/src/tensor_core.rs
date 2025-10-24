//! Core tensor definitions and traits.
//!
//! This module contains the fundamental building blocks of the tensor system:
//! - Core traits (AsAny, DifferentiableFunction, Function)
//! - Device enumeration and implementations
//! - Tensor struct definition

use core::any::Any;
use core::fmt;
use std::sync::Arc;

/// Extension trait for downcasting Function objects
///
/// This trait enables safe downcasting of trait objects to concrete types
/// without creating circular dependencies between tensor and autograd crates.
/// The autograd crate will provide implementations of this trait.
pub trait AsAny {
    /// Get as Any reference for downcasting
    fn as_any(&self) -> &dyn Any;
}

impl<B, S, T> AsAny for Tensor<B, S, T>
where
    B: Backend,
    S: Storage<T> + 'static,
    T: DataType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait DifferentiableFunction<B, S, T>: Send + Sync + fmt::Debug + AsAny
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Get the name of this function for debugging
    fn name(&self) -> &'static str;
}

/// Core trait for automatic differentiation functions
///
/// This trait defines the interface for functions that can participate in
/// the backward pass of automatic differentiation. It extends DifferentiableFunction
/// with methods needed for gradient computation.
///
/// # Generic Support
/// Fully generic over Backend<B>, Storage<S>, and DataType<T> for zero-cost abstractions.
pub trait Function<B, S, T>: DifferentiableFunction<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    /// Get references to the input tensors used in this operation
    ///
    /// # Returns
    /// Vector of Arc references to input tensors (for accessing values during backward)
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>];

    /// Compute gradients with respect to inputs given gradients w.r.t. outputs
    ///
    /// # Arguments
    /// * `grad_output` - Gradient tensor w.r.t. this function's output
    ///
    /// # Returns
    /// Vector of gradient tensors w.r.t. each input, in the same order as inputs
    fn backward(&self, grad_output: &Tensor<B, S, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>>;
}

// Re-exports for convenience
pub use coeus_backend::{Backend, CpuBackend};
pub use coeus_dtype::traits::FloatExt;
pub use coeus_dtype::DataType;
pub use coeus_storage::{DenseStorage, Shape, Storage, StorageFromVec, StorageToDense};

/// Compute device enumeration
///
/// Represents the different hardware backends available for tensor operations.
/// Each variant corresponds to a different compute substrate with different
/// performance characteristics and memory models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU device (default)
    #[default]
    Cpu,
    /// GPU device with index
    Gpu(usize),
    /// Neural Processing Unit
    Npu,
    /// Tensor Processing Unit
    Tpu,
}

impl Device {
    /// Get device name as string.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu(_) => "gpu",
            Self::Npu => "npu",
            Self::Tpu => "tpu",
        }
    }

    /// Check if device is CPU.
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Check if device is GPU.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    /// Get GPU index if device is GPU.
    #[must_use]
    pub fn gpu_index(&self) -> Option<usize> {
        match self {
            Self::Gpu(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu(idx) => write!(f, "gpu:{idx}"),
            Self::Npu => write!(f, "npu"),
            Self::Tpu => write!(f, "tpu"),
        }
    }
}

/// Core tensor type with nested backend/storage/dtype hierarchy.
///
/// # Type Parameters
///
/// - `B`: Backend implementation (e.g., `CpuBackend`)
/// - `S`: Storage implementation (e.g., `DenseStorage<T>`)
/// - `T`: Element data type (e.g., `Float32`)
///
/// # Safety
///
/// All operations are memory-safe. The type system ensures:
/// - No dtype mismatches at compile time
/// - No backend/storage incompatibility
/// - Thread-safe execution via `Send + Sync` bounds
///
/// # Examples
///
/// ```
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
/// use num_traits::Zero;
///
/// // Create zeros tensor
/// let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3]).unwrap();
/// assert_eq!(tensor.shape().dims(), &[2, 3]);
/// assert_eq!(tensor.len(), 6);
/// ```
#[derive(Debug)]
pub struct Tensor<B, S, T>
where
    B: Backend,
    S: Storage<T>,
    T: DataType,
{
    pub(crate) storage: S,
    pub(crate) backend: B,
    pub(crate) requires_grad: bool,
    /// Gradient tensor (None if not computed yet)
    /// Stored as Arc<RwLock<>> for thread-safe gradient accumulation
    #[cfg(feature = "std")]
    #[allow(clippy::type_complexity)]
    pub(crate) grad: Arc<std::sync::RwLock<Option<std::boxed::Box<Tensor<B, S, T>>>>>,
    #[cfg(not(feature = "std"))]
    pub(crate) grad: Arc<spin::RwLock<Option<alloc::boxed::Box<Tensor<B, S, T>>>>>,
    /// Function that created this tensor (for automatic differentiation)
    /// None if this tensor was created directly (leaf tensor)
    #[cfg(feature = "std")]
    pub(crate) grad_fn: Option<Arc<dyn Function<B, S, T>>>,
    #[cfg(not(feature = "std"))]
    pub(crate) grad_fn: Option<Arc<dyn Function<B, S, T>>>,
    pub(crate) _phantom: core::marker::PhantomData<T>,
}

// Implement Clone when all components are Clone
impl<B, S, T> Clone for Tensor<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone,
    T: DataType + Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            backend: self.backend.clone(),
            requires_grad: self.requires_grad,
            // Share gradient storage with the original tensor for autograd
            // This ensures gradients set on clones are visible to the original
            grad: self.grad.clone(),
            #[cfg(feature = "std")]
            grad_fn: self.grad_fn.clone(),
            #[cfg(not(feature = "std"))]
            grad_fn: self.grad_fn.clone(),
            _phantom: core::marker::PhantomData,
        }
    }
}
