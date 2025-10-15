//! Data transformation pipeline
//!
//! This module provides composable data transformations for preprocessing
//! machine learning data. Transformations can be chained together using
//! the `Compose` transform for complex preprocessing pipelines.
//!
//! ## Example
//!
//! ```rust
//! use coeus_utils::{ComposableTransform, Compose, Normalize, ToTensor};
//! use coeus_tensor::Tensor;
//! use coeus_backend::CpuBackend;
//! use coeus_storage::DenseStorage;
//! use coeus_dtype::float::Float32;
//!
//! // Create a transformation pipeline
//! let transform = Compose::new(vec![
//!     Box::new(ToTensor::new()),
//!     Box::new(Normalize::single_channel(2.0, 1.0)), // Normalize with mean=2.0, std=1.0
//! ]);
//!
//! // Apply transformations to data
//! let input = Box::new(vec![1.0, 2.0, 3.0, 4.0]); // Raw data
//! let transformed = transform.apply_dynamic(input).unwrap();
//!
//! // Result is a normalized tensor: (x - 2.0) / 1.0
//! let tensor = transformed.downcast::<Tensor<CpuBackend, DenseStorage<Float32>, Float32>>().unwrap();
//! ```

pub mod compose;
pub mod normalize;
pub mod totensor;

pub use compose::Compose;
pub use normalize::Normalize;
pub use totensor::ToTensor;

/// Trait for data transformations
///
/// Transformations convert raw data into processed tensors suitable for
/// neural network input. They are composable and can be chained together.
pub trait Transform<T, U = T> {
    /// Apply the transformation to input data
    ///
    /// # Arguments
    /// * `input` - The input data to transform
    ///
    /// # Returns
    /// The transformed data, or an error if transformation fails
    fn apply(&self, input: T) -> Result<U, TransformError>;
}

/// Errors that can occur during data transformation
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// Invalid input data format
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },

    /// Transformation-specific error
    #[error("Transformation error: {message}")]
    TransformError { message: String },

    /// Shape mismatch during transformation
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// Unsupported data type
    #[error("Unsupported data type: {type_name}")]
    UnsupportedType { type_name: String },

    /// Tensor operation error
    #[error("Tensor error: {0}")]
    TensorError(#[from] coeus_tensor::TensorError),
}
