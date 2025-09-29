//! Coeus Backend: Hierarchical GPU/CPU dispatch layer.
//!
//! Organized as a deep vertical dendrogram:
//! - traits/: Core interfaces (Backend, Ops)
//! - cpu/: CPU implementations (mod cpu;)
//! - gpu/: GPU implementations (mod gpu;)
//! - dispatch/: Runtime backend selection (mod dispatch;)
//!
//! Usage: use coeus_backend::dispatch::BackendKind; let backend = BackendKind::Gpu.create();

pub mod cpu;
// pub mod cuda; // Temporarily disabled - missing cuda_sys dependency
pub mod dispatch;
pub mod error;
// pub mod gpu; // Temporarily disabled - trait implementation issues
pub mod traits;
// pub mod tpu; // Temporarily disabled - missing tflite_sys dependency

pub use error::{BackendError, Result};
pub use traits::Backend;
pub use cpu::CpuBackend;
pub use dispatch::{BackendKind, select_backend};
// pub use cuda::CudaBackend; // Temporarily disabled
// pub use tpu::TpuBackend; // Temporarily disabled
// pub use gpu::GpuBackend; // Temporarily disabled

// Re-export common types
use std::ops::{Deref, DerefMut};

use coeus_dtype::Dtype;

/// BackendData: Enum for backend-specific data storage with shape
#[derive(Clone, Debug)]
pub enum BackendData<T: Dtype> {
    Cpu { data: Vec<T>, shape: Vec<usize> },
    Gpu { buffer: (), shape: Vec<usize> }, // Stub for GPU
}

impl<T: Dtype> BackendData<T> {
    pub fn cpu(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::Cpu { data, shape }
    }

    /// Returns a slice to the underlying data.
    pub fn data(&self) -> &[T] {
        match self {
            BackendData::Cpu { data, .. } => data,
            BackendData::Gpu { .. } => todo!("GPU data access"),
        }
    }

    /// Returns a slice to the shape vector.
    pub fn shape(&self) -> &[usize] {
        match self {
            BackendData::Cpu { shape, .. } => shape,
            BackendData::Gpu { shape, .. } => shape,
        }
    }

    /// Returns a mutable slice to the underlying data.
    pub fn data_mut(&mut self) -> &mut [T] {
        match self {
            BackendData::Cpu { data, .. } => data,
            BackendData::Gpu { .. } => todo!("GPU mutable data"),
        }
    }

    /// Returns the total number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns true if the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Dtype> Deref for BackendData<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        match self {
            BackendData::Cpu { data, .. } => data,
            BackendData::Gpu { .. } => panic!("GPU Deref not implemented"),
        }
    }
}

impl<T: Dtype> DerefMut for BackendData<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            BackendData::Cpu { data, .. } => data,
            BackendData::Gpu { .. } => panic!("GPU DerefMut not implemented"),
        }
    }
}

/// Device enum
#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
}
