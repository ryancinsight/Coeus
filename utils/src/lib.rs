//! # Coeus Utils
//!
//! PyTorch-style utility functions and data loading utilities for the Coeus tensor library.
//!
//! This crate provides high-level utilities that complement the core tensor operations,
//! including data loading, preprocessing, and common machine learning utilities.
//!
//! ## Features
//!
//! - **Data Loading**: PyTorch-compatible `Dataset` and `DataLoader` for efficient data iteration
//! - **Data Transformations**: Common preprocessing operations for tensors
//! - **Utilities**: Helper functions for common ML operations
//! - **Parallel Processing**: Efficient batch processing with configurable parallelism
//!
//! ## Examples
//!
//! ```rust
//! use coeus_utils::data::{Dataset, DataLoader};
//! use coeus_tensor::Tensor;
//!
//! // Create a simple dataset
//! struct MyDataset {
//!     data: Vec<Tensor<f32>>,
//!     targets: Vec<Tensor<f32>>,
//! }
//!
//! impl Dataset<f32> for MyDataset {
//!     fn len(&self) -> usize {
//!         self.data.len()
//!     }
//!
//!     fn get(&self, index: usize) -> (Tensor<f32>, Tensor<f32>) {
//!         (self.data[index].clone(), self.targets[index].clone())
//!     }
//! }
//!
//! // Create data loader (simplified example - multi-threading disabled due to current Tensor implementation)
//! let data = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])];
//! let targets = vec![Tensor::from_vec(vec![0.0], vec![1])];
//! // Note: DataLoader requires Send + Sync bounds for parallel loading
//! // which Tensor currently doesn't implement due to internal RefCell usage
//! // For single-threaded usage, implement your dataset accordingly
//! ```

pub mod data;
pub mod transforms;
pub mod utils;

pub use data::{DataLoader, DataLoaderBuilder, Dataset};
pub use utils::tensor_ops;
pub use utils::*;

// Re-export key types for convenience
pub use data::{ConcatDataset, Subset, TensorDataset};
pub use transforms::{
    ColorJitter,
    Compose,
    Identity,
    InterpolationMode,
    Lambda,
    Normalize,
    RandomAffine,
    // General transforms
    RandomApply,
    RandomChoice,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    RandomOrder,
    RandomPerspective,
    RandomRotation,
    // Vision transforms
    RandomVerticalFlip,
    ToTensor,
    // Core transforms
    Transform,
};

// Re-export utilities
pub use utils::{
    // Advanced losses
    advanced_loss::kl_div_loss,
    // Metrics
    metrics::{
        accuracy, auc_roc, classification_report, confusion_matrix, mean_squared_error,
        top_k_accuracy,
    },
    // Structures
    ClassificationReport,
    // Loss functions
    Reduction,
};

// Type alias for Result
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
