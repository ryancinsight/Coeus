//! Data transformation pipeline
//!
//! This module provides composable data transformations for preprocessing
//! machine learning data. Transformations can be chained together using
//! the `Compose` transform for complex preprocessing pipelines.
//!
//! ## PyO3 Safety Features
//!
//! The transform trait system is designed to be safely used with PyO3 trait objects:
//! - Zero-cost abstraction with static dispatch where possible
//! - Safe trait object composition through dynamic dispatch
//! - Memory safety guarantees across FFI boundaries
//! - SIMD acceleration for performance-critical operations
//!
//! ## Example
//!
//! ```rust
//! use utils::{ComposableTransform, Compose, Normalize, ToTensor};
//! use tensor::Tensor;
//! use backend::CpuBackend;
//! use storage::DenseStorage;
//! use dtype::float::Float32;
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
//! let tensor = transformed.downcast::<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>().unwrap();
//! ```

use backend::Backend;
use dtype::DataType;
use storage::Storage;

pub mod compose;
pub mod normalize;
pub mod random_apply;
pub mod resize;
pub mod totensor;

pub use compose::Compose;
pub use normalize::Normalize;
pub use random_apply::{ConditionalTransform, RandomApply};
pub use resize::Resize;
pub use totensor::ToTensor;

/// Core trait for data transformations that work with generic tensor types
///
/// This trait supports the full B<S<T>> generic architecture and enables
/// zero-cost abstractions across different backend, storage, and data type combinations.
pub trait CoeusTransform<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Apply the transformation to a tensor with the specified types
    ///
    /// # Arguments
    /// * `input` - The input tensor to transform
    ///
    /// # Returns
    /// The transformed tensor, or an error if transformation fails
    fn apply(
        &self,
        input: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<coeus_tensor::Tensor<B, S, T>, TransformError>;

    /// Get the name of this transform for debugging
    fn name(&self) -> &str;
}

/// Legacy trait for data transformations (deprecated)
///
/// Transformations convert raw data into processed tensors suitable for
/// neural network input. They are composable and can be chained together.
///
/// This trait is maintained for backward compatibility but should be replaced
/// with CoeusTransform for new implementations.
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
