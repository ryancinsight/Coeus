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
    B: Backend<Data = T>,
    S: Storage<T> + 'static,
    T: DataType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait DifferentiableFunction<B, S, T>: Send + Sync + fmt::Debug + AsAny
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Get the name of this function for debugging
    fn name(&self) -> &'static str;
}

/// Core function trait for automatic differentiation operations
///
/// This trait defines the interface for differentiable operations that can
/// participate in the computation graph and gradient computation.
///
/// # Type Parameters
/// * `B` - Backend type
/// * `S` - Storage type for inputs
/// * `T` - Data type
pub trait Function<B, S, T>: DifferentiableFunction<B, S, T>
where
    B: Backend<Data = T>,
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
    /// * `grad_output` - Gradient tensor w.r.t. this function's output (dense)
    ///
    /// # Returns
    /// Vector of gradient tensors w.r.t. each input, in the same order as inputs.
    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> anyhow::Result<Vec<Tensor<B, S, T>>>;
}


// Re-exports for convenience
pub use backend::{Backend, CpuBackend};
pub use dtype::traits::FloatExt;
pub use dtype::DataType;
pub use storage::{DenseStorage, Shape, Storage, StorageFromVec, StorageToDense};

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

pub struct Tensor<B, S, T>
where
    B: Backend<Data = T>,
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
    pub(crate) grad_fn: Option<String>,
}

// Implement Clone when all components are Clone
impl<B, S, T> Clone for Tensor<B, S, T>
where
    B: Backend<Data = T> + Clone,
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
        }
    }
}

impl<B, S, T> fmt::Debug for Tensor<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.storage.shape())
            .field("len", &self.storage.len())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &true) // Simplified for Debug - checking RwLock would require locking
            .finish()
    }
}
