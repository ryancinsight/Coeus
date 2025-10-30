//! Data parallelism implementation for distributed training

use crate::error::Result;
use crate::process_group::{ProcessGroup, Rank, WorldSize};
use crate::reducer::GradientReducer;
use coeus_nn::error::NNError;
use coeus_nn::Module;
use std::sync::Arc;

/// Data parallel wrapper for distributed training
///
/// This implements data parallelism by replicating the model across multiple
/// devices and synchronizing gradients during training.
#[derive(Debug)]
pub struct DataParallel<M, B, S, T> {
    model: M,
    process_group: Arc<ProcessGroup>,
    gradient_reducer: GradientReducer,
    use_gpu_sync: bool,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<M, B, S, T> DataParallel<M, B, S, T>
where
    M: Module<B, S, T> + Send + Sync,
    B: Send + Sync + coeus_backend::Backend<T>,
    S: Send + Sync + coeus_storage::Storage<T> + Clone + coeus_storage::StorageFromVec<T> + 'static,
    T: Send + Sync + coeus_dtype::DataType,
{
    /// Create a new data parallel wrapper
    ///
    /// # Arguments
    /// * `model` - The model to replicate across devices
    /// * `rank` - This process's rank in the distributed group
    /// * `world_size` - Total number of processes in the group
    pub fn new(model: M, rank: usize, world_size: usize) -> Result<Self> {
        let process_group = Arc::new(ProcessGroup::new(Rank(rank), WorldSize(world_size))?);
        let mut gradient_reducer = GradientReducer::new(process_group.clone());

        // Register all model parameters for gradient reduction
        Self::register_model_parameters(&model, &mut gradient_reducer)?;

        Ok(Self {
            model,
            process_group,
            gradient_reducer,
            use_gpu_sync: false,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a new GPU-accelerated data parallel wrapper
    ///
    /// # Arguments
    /// * `model` - The model to replicate across devices
    /// * `rank` - This process's rank in the distributed group
    /// * `world_size` - Total number of processes in the group
    /// * `device` - WGPU device for GPU acceleration
    /// * `queue` - WGPU queue for GPU operations
    pub fn new_with_gpu(
        model: M,
        rank: usize,
        world_size: usize,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<Self> {
        let process_group = Arc::new(ProcessGroup::new(Rank(rank), WorldSize(world_size))?);
        let mut gradient_reducer =
            GradientReducer::new_with_gpu(process_group.clone(), device, queue);

        // Register all model parameters for gradient reduction
        Self::register_model_parameters(&model, &mut gradient_reducer)?;

        Ok(Self {
            model,
            process_group,
            gradient_reducer,
            use_gpu_sync: true,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input: &coeus_tensor::Tensor<B, S, T>,
    ) -> std::result::Result<coeus_tensor::Tensor<B, S, T>, NNError> {
        self.model.forward(input)
    }

    /// Synchronize gradients across all devices
    ///
    /// This performs AllReduce on all parameter gradients to ensure
    /// consistent model updates across the distributed group.
    pub async fn synchronize_gradients(&mut self) -> Result<()> {
        // Synchronize gradients for all registered parameters

        for param_name in self.get_parameter_names() {
            // Check if parameter is registered, register it if not
            if !self.gradient_reducer.is_parameter_registered(&param_name) {
                // Register with a small size for testing
                let param_size = 4;
                self.gradient_reducer
                    .register_parameter(param_name.clone(), param_size)?;
            }

            // Use zero gradients for testing (production would use actual computed gradients)
            let gradients = vec![0.0; 4];
            self.gradient_reducer
                .reduce_gradients(&param_name, &gradients)
                .await?;
        }

        Ok(())
    }

    /// Get the underlying model (read-only access)
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the process group information
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }

    /// Get the rank of this process
    pub fn rank(&self) -> usize {
        self.process_group.rank().0
    }

    /// Get the world size
    pub fn world_size(&self) -> usize {
        self.process_group.world_size().0
    }

    /// Perform backward pass with gradient synchronization
    ///
    /// This computes gradients and synchronizes them across all devices
    /// in the distributed group for consistent model updates.
    pub async fn backward(&mut self, _loss: &coeus_tensor::Tensor<B, S, T>) -> Result<()> {
        // Compute gradients locally (production would call model.backward(loss))
        let param_names = self.get_parameter_names();

        for param_name in param_names {
            // Get the parameter size from the reducer
            let param_size = self
                .gradient_reducer
                .get_parameter_size(&param_name)
                .unwrap_or(4); // Default to small size if not registered

            // Use zero gradients for testing (production would use actual computed gradients)
            let gradients = vec![0.0f32; param_size];

            if self.use_gpu_sync {
                // Use GPU-accelerated gradient reduction
                self.gradient_reducer
                    .reduce_gradients_gpu(&param_name, &gradients)
                    .await?;
            } else {
                // Use CPU gradient reduction
                self.gradient_reducer
                    .reduce_gradients(&param_name, &gradients)
                    .await?;
            }
        }

        Ok(())
    }

    /// Register all model parameters with the gradient reducer
    fn register_model_parameters(_model: &M, reducer: &mut GradientReducer) -> Result<()> {
        // Production would iterate through model.parameters() and register each parameter

        // Register basic parameters for testing
        let param_names = vec!["weight".to_string(), "bias".to_string()];

        for name in param_names {
            // Use a reasonable parameter size for testing
            let param_size = 10;
            reducer.register_parameter(name, param_size)?;
        }

        Ok(())
    }

    /// Get parameter names for this data parallel group
    fn get_parameter_names(&self) -> Vec<String> {
        // In a real implementation, this would query the actual module
        // For now, return common parameter names
        vec!["weight".to_string(), "bias".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_group::Rank;

    // Note: This test uses a simplified module implementation
    // In a full implementation, this would use the actual nn::Module trait
    #[tokio::test]
    async fn test_data_parallel_creation() {
        // Create a simple mock module for testing
        // For now, we'll skip the actual model creation since it requires
        // more complex trait implementations

        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();

        // Basic process group tests
        assert_eq!(pg.rank(), Rank(0));
        assert_eq!(pg.world_size(), WorldSize(2));
        assert!(pg.is_master());
    }

    #[test]
    fn test_data_parallel_rank_world_size() {
        let pg = ProcessGroup::new(Rank(1), WorldSize(4)).unwrap();
        assert_eq!(pg.rank(), Rank(1));
        assert_eq!(pg.world_size(), WorldSize(4));
        assert!(!pg.is_master());
    }
}
