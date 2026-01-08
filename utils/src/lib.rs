//! # Coeus Utils
//!
//! Data loading utilities and datasets for Coeus, providing PyTorch-compatible
//! dataset and dataloader APIs for efficient machine learning workflows.
//!
//! ## Features
//!
//! - **Dataset Trait**: PyTorch-compatible dataset abstraction
//! - **DataLoader**: Iterator-based batching and shuffling
//! - **Samplers**: Control data access patterns (sequential, random, batched)
//! - **Data Transforms**: Composable preprocessing pipelines (ToTensor, Normalize, Compose)
//! - **Common Datasets**: MNIST, CIFAR-10, TensorDataset, ImageFolder
//! - **Memory Safety**: Zero unsafe code, ownership-based data access
//! - **Performance**: Iterator-based design with zero-cost abstractions
//!
//! ## Example
//!
//! ```rust
//! use utils::{Dataset, DataLoader, TensorDataset};
//! use tensor::Tensor;
//! use backend::CpuBackend;
//! use storage::DenseStorage;
//! use dtype::float::Float32;
//! use dtype::int::Int32;
//!
//! // Create a simple dataset from tensors
//! let data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
//!     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
//!     &[4]
//! ).unwrap();
//! let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
//!     vec![Int32::new(0), Int32::new(1), Int32::new(0), Int32::new(1)],
//!     &[4]
//! ).unwrap();
//! let dataset = TensorDataset::new(vec![data], vec![targets]).unwrap();
//!
//! // Create a data loader with batching
//! let dataloader = DataLoader::builder(dataset)
//!     .batch_size(2)
//!     .shuffle(true)
//!     .build().unwrap();
//!
//! // Iterate over batches
//! for batch_result in dataloader {
//!     let batch = batch_result.unwrap();
//!     println!("Batch size: {}", batch.len());
//!     // Train your model...
//! }
//! ```

#[cfg(test)]
mod integration_tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::{float::Float32, int::Int32};
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_dataloader_nn_integration() {
        // Create synthetic training data
        let inputs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.1),
                Float32::new(0.2),
                Float32::new(0.3),
                Float32::new(0.4),
                Float32::new(0.5),
                Float32::new(0.6),
                Float32::new(0.7),
                Float32::new(0.8),
            ],
            &[8],
        )
        .unwrap();

        let targets = Tensor::<CpuBackend<Int32>, DenseStorage<Int32>, Int32>::from_vec(
            vec![
                Int32::new(0),
                Int32::new(1),
                Int32::new(0),
                Int32::new(1),
                Int32::new(0),
                Int32::new(1),
                Int32::new(0),
                Int32::new(1),
            ],
            &[8],
        )
        .unwrap();

        // Create dataset
        let dataset = TensorDataset::new(vec![inputs], vec![targets]).unwrap();
        assert_eq!(dataset.len(), 8);

        // Create data loader with batching
        let dataloader = DataLoader::builder(dataset)
            .batch_size(4)
            .shuffle(false) // Deterministic for testing
            .build()
            .unwrap();

        // Collect all batches
        let mut batches = Vec::new();
        for batch_result in dataloader {
            let batch = batch_result.unwrap();
            batches.push(batch);
        }

        // Should have 2 batches of size 4 each
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 4);
        assert_eq!(batches[1].len(), 4);

        // Verify batch structure (each sample has 1 input tensor and 1 target tensor)
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.inputs.len(), 1); // 1 input tensor
                assert_eq!(sample.targets.len(), 1); // 1 target tensor
                assert_eq!(sample.inputs[0].shape().dims(), &[1]); // Scalar input
                assert_eq!(sample.targets[0].shape().dims(), &[1]); // Scalar target
            }
        }

        println!("DataLoader integration test passed - ready for neural network training!");
    }
}

pub mod dataloader;
pub mod dataset;
pub mod error;
pub mod sampler;
pub mod transforms;

pub mod datasets {
    pub mod tensor;
}

pub use datasets::tensor::{ConcatDataset, Subset, TensorDataset, TensorSample};

pub use dataloader::DataLoader;
pub use dataset::{Dataset, DatasetExt};
pub use error::DataError;
pub use sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};
pub use transforms::compose::{ComposableTransform, SimdCompose};
pub use transforms::random_apply::{ConditionalTransform, RandomApply};
pub use transforms::{Compose, Transform, TransformError};
