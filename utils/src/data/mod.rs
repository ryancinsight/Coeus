//! Data loading utilities with PyTorch-compatible API
//!
//! This module provides `Dataset` and `DataLoader` traits and implementations
//! that mirror PyTorch's data loading functionality for seamless migration.

pub mod dataloader;
pub mod dataset;

pub use dataloader::{Batch, DataLoader, DataLoaderBuilder};
pub use dataset::{ConcatDataset, Dataset, Subset, TensorDataset};
