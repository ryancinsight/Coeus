//! Backend errors.

use thiserror::Error;

/// Backend errors.
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        message: String,
    },
    #[error("Device error on {device}")]
    DeviceError {
        device: String,
    },
    #[error("Invalid dimension {0}")]
    InvalidDimension(usize),
    #[error("Out of bounds index {0}")]
    OutOfBounds(i64),
    #[error("GPU device creation failed: {0}")]
    GpuDevice(String),
    #[error("CPU operation failed: {0}")]
    Cpu(String),
    #[error("Fallback from GPU to CPU failed")]
    Fallback,
    #[error("DLL version mismatch: {0}")]
    DllVersionMismatch(String),
    #[error("Generic dtype mismatch")]
    DtypeMismatch,
    #[error("Concurrency error: {0}")]
    Concurrency(String),
    #[error("CUDA load failed: {0}")]
    CudaLoad(String),
    #[error("cuDNN error: {0}")]
    Cudnn(String),
    #[error("Vulkan pool exhausted")]
    PoolExhausted,
    #[error("Stream lifetime error: {0}")]
    StreamLifetime(String),
    #[error("ONNX serialization error: {0}")]
    OnnxSerialize(String),
    #[error("Quantization overflow: {0}")]
    QuantOverflow(String),
    #[error("TPU inference error: {0}")]
    TpuInfer(String),
    #[error("FP16 overflow: {0}")]
    Fp16Overflow(String),
    #[error("Activation NaN input")]
    ActivationNaN,
    #[error("Layer NaN output")]
    LayerNaN,
    #[error("Transformer layer NaN output")]
    TransformerNaN,
    #[error("Advanced NN mechanism mismatch: {0}")]
    MechanismMismatch(String),
    #[error("Training op mismatch: {0}")]
    TrainingMismatch(String),
    #[error("Optimizer NaN update")]
    OptimizerNaN,
    #[error("Backward NaN gradient")]
    BackwardNaN,
    #[error("Refinement NaN gradient")]
    RefineNaN,
    #[error("Distributed op mismatch: {0}")]
    DistributedMismatch(String),
    #[error("Fused operation NaN output")]
    FusedNaN,
    #[error("Unsupported operation")]
    UnsupportedOperation,

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Result type for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

impl Clone for BackendError {
    fn clone(&self) -> Self {
        match self {
            BackendError::ShapeMismatch { expected, actual } => BackendError::ShapeMismatch { expected: expected.clone(), actual: actual.clone() },
            BackendError::InvalidOperation { message } => BackendError::InvalidOperation { message: message.clone() },
            BackendError::DeviceError { device } => BackendError::DeviceError { device: device.clone() },
            BackendError::GpuDevice(s) => BackendError::GpuDevice(s.clone()),
            BackendError::Cpu(s) => BackendError::Cpu(s.clone()),
            BackendError::Fallback => BackendError::Fallback,
            BackendError::DllVersionMismatch(s) => BackendError::DllVersionMismatch(s.clone()),
            BackendError::DtypeMismatch => BackendError::DtypeMismatch,
            BackendError::Concurrency(s) => BackendError::Concurrency(s.clone()),
            BackendError::CudaLoad(s) => BackendError::CudaLoad(s.clone()),
            BackendError::Cudnn(s) => BackendError::Cudnn(s.clone()),
            BackendError::PoolExhausted => BackendError::PoolExhausted,
            BackendError::StreamLifetime(s) => BackendError::StreamLifetime(s.clone()),
            BackendError::OnnxSerialize(s) => BackendError::OnnxSerialize(s.clone()),
            BackendError::QuantOverflow(s) => BackendError::QuantOverflow(s.clone()),
            BackendError::TpuInfer(s) => BackendError::TpuInfer(s.clone()),
            BackendError::Fp16Overflow(s) => BackendError::Fp16Overflow(s.clone()),
            BackendError::ActivationNaN => BackendError::ActivationNaN,
            BackendError::LayerNaN => BackendError::LayerNaN,
            BackendError::TransformerNaN => BackendError::TransformerNaN,
            BackendError::MechanismMismatch(s) => BackendError::MechanismMismatch(s.clone()),
            BackendError::TrainingMismatch(s) => BackendError::TrainingMismatch(s.clone()),
            BackendError::OptimizerNaN => BackendError::OptimizerNaN,
            BackendError::BackwardNaN => BackendError::BackwardNaN,
            BackendError::RefineNaN => BackendError::RefineNaN,
            BackendError::DistributedMismatch(s) => BackendError::DistributedMismatch(s.clone()),
            BackendError::FusedNaN => BackendError::FusedNaN,
            BackendError::UnsupportedOperation => BackendError::UnsupportedOperation,
            BackendError::NotImplemented(s) => BackendError::NotImplemented(s.clone()),
            BackendError::InvalidDimension(dim) => BackendError::InvalidDimension(*dim),
            BackendError::OutOfBounds(idx) => BackendError::OutOfBounds(*idx),
        }
    }
}

impl BackendError {
    /// Create a new GPU error with message
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuDevice(message.into())
    }

    /// Create a new allocation error with message
    pub fn allocation_error<S: Into<String>>(message: S) -> Self {
        Self::Cpu(format!("Allocation error: {}", message.into()))
    }

    /// Create a new invalid operation error with message
    pub fn invalid_operation<S: Into<String>>(message: S) -> Self {
        Self::InvalidOperation { message: message.into() }
    }
}

