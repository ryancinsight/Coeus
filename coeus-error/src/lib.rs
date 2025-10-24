//! Unified Error Types for Coeus
//!
//! This crate provides a unified error hierarchy for the entire Coeus deep learning
//! framework. All crates should use these error types for consistency and proper
//! error propagation across the stack.

#![no_std]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::string::String;
use core::fmt;

/// Result type alias using the unified error type
pub type Result<T> = core::result::Result<T, Error>;

/// Unified error type for the entire Coeus framework
///
/// This enum consolidates all error types from across the codebase into a single,
/// hierarchical error type. Each variant represents a different subsystem or
/// category of errors with optional source error chaining for better debugging.
///
/// # Error Hierarchy
///
/// ```text
/// Error
/// ├── Backend
/// │   ├── Cpu
/// │   ├── Gpu
/// │   ├── Tpu
/// │   └── Npu
/// ├── Storage
/// │   ├── Dense
/// │   ├── Sparse
/// │   └── Quantized
/// ├── Tensor
/// │   ├── Shape
/// │   ├── DataType
/// │   └── Operation
/// ├── Autograd
/// │   ├── Graph
/// │   ├── Gradient
/// │   └── Function
/// ├── NN
/// │   ├── Module
/// │   ├── Layer
/// │   ├── Training
/// │   └── Serialization
/// ├── Optimizer
/// │   ├── SGD
/// │   ├── Adam
/// │   ├── RMSprop
/// │   └── Config
/// ├── Distributed
/// │   ├── ProcessGroup
/// │   ├── Communication
/// │   └── Synchronization
/// ├── IO
/// │   ├── Serialization
/// │   ├── Deserialization
/// │   └── FileSystem
/// └── System
///     ├── Memory
///     ├── Threading
///     └── Platform
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Backend-related errors (CPU, GPU, TPU, NPU)
    Backend(BackendError),
    /// Storage-related errors (Dense, Sparse, Quantized)
    Storage(StorageError),
    /// Tensor-related errors (shape, dtype, operations)
    Tensor(TensorError),
    /// Automatic differentiation errors
    Autograd(AutogradError),
    /// Neural network errors
    NN(NNError),
    /// Optimizer errors
    Optimizer(OptimizerError),
    /// Distributed training errors
    Distributed(DistributedError),
    /// I/O and serialization errors
    IO(IOError),
    /// System-level errors (memory, threading, platform)
    System(SystemError),
}

/// Backend subsystem errors
#[derive(Debug)]
#[non_exhaustive]
pub enum BackendError {
    /// CPU backend errors
    Cpu(String),
    /// GPU backend errors
    Gpu(String),
    /// TPU backend errors
    Tpu(String),
    /// NPU backend errors
    Npu(String),
    /// Device not found
    DeviceNotFound(String),
    /// Operation not supported
    OperationNotSupported(String),
}

/// Storage subsystem errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StorageError {
    /// Dense storage errors
    Dense(String),
    /// Sparse storage errors
    Sparse(String),
    /// Quantized storage errors
    Quantized(String),
    /// Invalid shape
    InvalidShape(String),
    /// Memory allocation failed
    AllocationFailed(String),
}

/// Tensor subsystem errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorError {
    /// Shape mismatch errors
    ShapeMismatch(String),
    /// Data type mismatch errors
    DataTypeMismatch(String),
    /// Operation failed
    OperationFailed(String),
    /// Invalid index
    InvalidIndex(String),
    /// Broadcasting failed
    BroadcastingFailed(String),
}

/// Automatic differentiation errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AutogradError {
    /// Computation graph errors
    Graph(String),
    /// Gradient computation errors
    Gradient(String),
    /// Function definition errors
    Function(String),
    /// Variable tracking errors
    Variable(String),
}

/// Neural network errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NNError {
    /// Module errors
    Module(String),
    /// Layer configuration errors
    Layer(String),
    /// Training errors
    Training(String),
    /// Serialization errors
    Serialization(String),
    /// Invalid parameter
    InvalidParameter(String),
}

/// Optimizer errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OptimizerError {
    /// SGD optimizer errors
    SGD(String),
    /// Adam optimizer errors
    Adam(String),
    /// `RMSprop` optimizer errors
    RMSprop(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Parameter errors
    Parameter(String),
}

/// Distributed training errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DistributedError {
    /// Process group errors
    ProcessGroup(String),
    /// Communication errors
    Communication(String),
    /// Synchronization errors
    Synchronization(String),
    /// Rank mismatch
    RankMismatch(String),
}

/// I/O and serialization errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IOError {
    /// Serialization failed
    Serialization(String),
    /// Deserialization failed
    Deserialization(String),
    /// File system errors
    FileSystem(String),
    /// Invalid format
    InvalidFormat(String),
}

/// System-level errors
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SystemError {
    /// Memory allocation errors
    Memory(String),
    /// Threading errors
    Threading(String),
    /// Platform-specific errors
    Platform(String),
    /// Resource exhaustion
    ResourceExhaustion(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Backend(e) => write!(f, "Backend error: {e}"),
            Error::Storage(e) => write!(f, "Storage error: {e}"),
            Error::Tensor(e) => write!(f, "Tensor error: {e}"),
            Error::Autograd(e) => write!(f, "Autograd error: {e}"),
            Error::NN(e) => write!(f, "Neural network error: {e}"),
            Error::Optimizer(e) => write!(f, "Optimizer error: {e}"),
            Error::Distributed(e) => write!(f, "Distributed error: {e}"),
            Error::IO(e) => write!(f, "I/O error: {e}"),
            Error::System(e) => write!(f, "System error: {e}"),
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Cpu(msg) => write!(f, "CPU backend: {msg}"),
            BackendError::Gpu(msg) => write!(f, "GPU backend: {msg}"),
            BackendError::Tpu(msg) => write!(f, "TPU backend: {msg}"),
            BackendError::Npu(msg) => write!(f, "NPU backend: {msg}"),
            BackendError::DeviceNotFound(msg) => write!(f, "Device not found: {msg}"),
            BackendError::OperationNotSupported(msg) => write!(f, "Operation not supported: {msg}"),
        }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Dense(msg) => write!(f, "Dense storage: {msg}"),
            StorageError::Sparse(msg) => write!(f, "Sparse storage: {msg}"),
            StorageError::Quantized(msg) => write!(f, "Quantized storage: {msg}"),
            StorageError::InvalidShape(msg) => write!(f, "Invalid shape: {msg}"),
            StorageError::AllocationFailed(msg) => write!(f, "Allocation failed: {msg}"),
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            TensorError::DataTypeMismatch(msg) => write!(f, "Data type mismatch: {msg}"),
            TensorError::OperationFailed(msg) => write!(f, "Operation failed: {msg}"),
            TensorError::InvalidIndex(msg) => write!(f, "Invalid index: {msg}"),
            TensorError::BroadcastingFailed(msg) => write!(f, "Broadcasting failed: {msg}"),
        }
    }
}

impl fmt::Display for AutogradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutogradError::Graph(msg) => write!(f, "Computation graph: {msg}"),
            AutogradError::Gradient(msg) => write!(f, "Gradient computation: {msg}"),
            AutogradError::Function(msg) => write!(f, "Function definition: {msg}"),
            AutogradError::Variable(msg) => write!(f, "Variable tracking: {msg}"),
        }
    }
}

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NNError::Module(msg) => write!(f, "Module: {msg}"),
            NNError::Layer(msg) => write!(f, "Layer: {msg}"),
            NNError::Training(msg) => write!(f, "Training: {msg}"),
            NNError::Serialization(msg) => write!(f, "Serialization: {msg}"),
            NNError::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
        }
    }
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::SGD(msg) => write!(f, "SGD optimizer: {msg}"),
            OptimizerError::Adam(msg) => write!(f, "Adam optimizer: {msg}"),
            OptimizerError::RMSprop(msg) => write!(f, "`RMSprop` optimizer: {msg}"),
            OptimizerError::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            OptimizerError::Parameter(msg) => write!(f, "Parameter: {msg}"),
        }
    }
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributedError::ProcessGroup(msg) => write!(f, "Process group: {msg}"),
            DistributedError::Communication(msg) => write!(f, "Communication: {msg}"),
            DistributedError::Synchronization(msg) => write!(f, "Synchronization: {msg}"),
            DistributedError::RankMismatch(msg) => write!(f, "Rank mismatch: {msg}"),
        }
    }
}

impl fmt::Display for IOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IOError::Serialization(msg) => write!(f, "Serialization: {msg}"),
            IOError::Deserialization(msg) => write!(f, "Deserialization: {msg}"),
            IOError::FileSystem(msg) => write!(f, "File system: {msg}"),
            IOError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
        }
    }
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemError::Memory(msg) => write!(f, "Memory: {msg}"),
            SystemError::Threading(msg) => write!(f, "Threading: {msg}"),
            SystemError::Platform(msg) => write!(f, "Platform: {msg}"),
            SystemError::ResourceExhaustion(msg) => write!(f, "Resource exhaustion: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl std::error::Error for BackendError {}

#[cfg(feature = "std")]
impl std::error::Error for StorageError {}

#[cfg(feature = "std")]
impl std::error::Error for TensorError {}

#[cfg(feature = "std")]
impl std::error::Error for AutogradError {}

#[cfg(feature = "std")]
impl std::error::Error for NNError {}

#[cfg(feature = "std")]
impl std::error::Error for OptimizerError {}

#[cfg(feature = "std")]
impl std::error::Error for DistributedError {}

#[cfg(feature = "std")]
impl std::error::Error for IOError {}

#[cfg(feature = "std")]
impl std::error::Error for SystemError {}

// From implementations for ergonomic error conversion
impl From<BackendError> for Error {
    fn from(err: BackendError) -> Self {
        Error::Backend(err)
    }
}

impl From<StorageError> for Error {
    fn from(err: StorageError) -> Self {
        Error::Storage(err)
    }
}

impl From<TensorError> for Error {
    fn from(err: TensorError) -> Self {
        Error::Tensor(err)
    }
}

impl From<AutogradError> for Error {
    fn from(err: AutogradError) -> Self {
        Error::Autograd(err)
    }
}

impl From<NNError> for Error {
    fn from(err: NNError) -> Self {
        Error::NN(err)
    }
}

impl From<OptimizerError> for Error {
    fn from(err: OptimizerError) -> Self {
        Error::Optimizer(err)
    }
}

impl From<DistributedError> for Error {
    fn from(err: DistributedError) -> Self {
        Error::Distributed(err)
    }
}

impl From<IOError> for Error {
    fn from(err: IOError) -> Self {
        Error::IO(err)
    }
}

impl From<SystemError> for Error {
    fn from(err: SystemError) -> Self {
        Error::System(err)
    }
}

// String conversion implementations for easier error creation
impl From<String> for BackendError {
    fn from(msg: String) -> Self {
        BackendError::Cpu(msg) // Default to CPU for generic string errors
    }
}

impl From<String> for StorageError {
    fn from(msg: String) -> Self {
        StorageError::Dense(msg) // Default to dense for generic string errors
    }
}

impl From<String> for TensorError {
    fn from(msg: String) -> Self {
        TensorError::OperationFailed(msg) // Default to operation failed for generic errors
    }
}

impl From<String> for AutogradError {
    fn from(msg: String) -> Self {
        AutogradError::Graph(msg) // Default to graph for generic autograd errors
    }
}

impl From<String> for NNError {
    fn from(msg: String) -> Self {
        NNError::Module(msg) // Default to module for generic NN errors
    }
}

impl From<String> for OptimizerError {
    fn from(msg: String) -> Self {
        OptimizerError::Parameter(msg) // Default to parameter for generic optimizer errors
    }
}

impl From<String> for DistributedError {
    fn from(msg: String) -> Self {
        DistributedError::Communication(msg) // Default to communication for generic distributed errors
    }
}

impl From<String> for IOError {
    fn from(msg: String) -> Self {
        IOError::Serialization(msg) // Default to serialization for generic I/O errors
    }
}

impl From<String> for SystemError {
    fn from(msg: String) -> Self {
        SystemError::Memory(msg) // Default to memory for generic system errors
    }
}

// Conversion implementations from existing error types
// These will be added as other crates are updated to use the unified error type

#[cfg(test)]
mod tests {
    use super::*;
    use crate::std::string::ToString;

    #[test]
    fn test_error_display() {
        let error = Error::Tensor(TensorError::ShapeMismatch("Test shape error".to_string()));
        assert!(error.to_string().contains("Tensor error"));
        assert!(error.to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_error_hierarchy() {
        // Test that we can nest errors properly
        let backend_error = BackendError::Gpu("GPU memory allocation failed".to_string());
        let error = Error::Backend(backend_error);
        assert!(matches!(error, Error::Backend(_)));
    }

    #[test]
    fn test_error_from_conversions() {
        // Test From implementations work correctly
        let tensor_err = TensorError::ShapeMismatch("shape mismatch".to_string());
        let error: Error = tensor_err.into();
        assert!(matches!(error, Error::Tensor(_)));

        // Test string conversion
        let str_err: TensorError = "operation failed".to_string().into();
        assert!(matches!(str_err, TensorError::OperationFailed(_)));
    }
}
