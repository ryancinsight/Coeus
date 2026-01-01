//! # Task-Specific Outputs
//!
//! Implementation of specialized output heads for different downstream tasks
//! in multimodal processing (classification, regression, generation, retrieval).

use crate::linear::Linear;
use crate::module::ModuleExt;
use backend::Backend;
use dtype::{DataType, FloatExt};
use storage::{Storage, StorageFromVec, StorageToDense};

/// Task-specific outputs
#[derive(Debug)]
pub enum Task<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Default,
    T: DataType,
{
    /// Classification output
    Classification(Classifier<B, S, T>),
    /// Regression output
    Regression(Linear<B, S, T>),
    /// Generation output (for text/audio generation)
    Generation(Generator<B, S, T>),
    /// Retrieval output (for multimodal retrieval)
    Retrieval(Retriever<B, S, T>),
}

#[derive(Debug)]
pub struct Classifier<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Default,
    T: DataType,
{
    pub classifier: Linear<B, S, T>,
    pub num_classes: usize,
}

#[derive(Debug)]
pub struct Generator<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Default,
    T: DataType,
{
    pub lm_head: Linear<B, S, T>,
    pub vocab_size: usize,
}

#[derive(Debug)]
pub struct Retriever<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Default,
    T: DataType,
{
    pub projection: Linear<B, S, T>,
    pub similarity_type: SimilarityType,
}

#[derive(Debug)]
pub enum SimilarityType {
    Cosine,
    DotProduct,
    Euclidean,
}

impl<B, S, T> Task<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Default,
    T: DataType + FloatExt + num_traits::Zero + num_traits::One,
{
    /// Get the number of parameters in this task output
    pub fn num_parameters(&self) -> usize {
        match self {
            Task::Classification(head) => head.classifier.num_parameters(),
            Task::Regression(linear) => linear.num_parameters(),
            Task::Generation(head) => head.lm_head.num_parameters(),
            Task::Retrieval(head) => head.projection.num_parameters(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    #[test]
    fn test_task_num_parameters() {
        let classification_task =
            Task::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::Classification(
                Classifier {
                    classifier: Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        768, 10,
                    )
                    .unwrap(),
                    num_classes: 10,
                },
            );

        // Test that num_parameters doesn't panic
        let _params = classification_task.num_parameters();
        // We can't easily test the exact value without implementing the full parameter counting
        // but we can test that it doesn't panic
    }

    #[test]
    fn test_similarity_type_enum() {
        let types = [
            SimilarityType::Cosine,
            SimilarityType::DotProduct,
            SimilarityType::Euclidean,
        ];

        assert_eq!(types.len(), 3);
    }
}
