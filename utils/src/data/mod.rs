//! Dataset implementations and utilities
//!
//! Provides PyTorch-compatible dataset functionality with automatic differentiation support.

pub mod concat;
pub mod dataloader;
pub mod dataset_trait;
pub mod subset;
pub mod tensor_dataset;

// Re-export for convenience
pub use concat::ConcatDataset;
pub use dataloader::{DataLoader, DataLoaderBuilder, DataLoaderIter};
pub use dataset_trait::{Dataset, DatasetIter};
pub use subset::Subset;
pub use tensor_dataset::TensorDataset;
