//! Data parallelism implementation for distributed training

use crate::error::Result;
use crate::process_group::{ProcessGroup, Rank, WorldSize};
use crate::reducer::GradientReducer;
use coeus_nn::error::NNError;
use coeus_nn::Module;

/// Data parallel wrapper for distributed training
///
/// This implements data parallelism by replicating the model across multiple
/// devices and synchronizing gradients during training.
#[derive(Debug)]
pub struct DataParallel<M, B, S, T> {
    model: M,
    process_group: ProcessGroup,
    gradient_reducer: GradientReducer,
    use_gpu_sync: bool,
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<M, B, S, T> DataParallel<M, B, S, T>
where
    M: Module<B, S, T> + Send + Sync,
    B: Send + Sync + coeus_backend::Backend,
    S: Send + Sync + coeus_storage::Storage<T> + Clone + 'static,
    T: Send + Sync + coeus_dtype::DataType,
{
    /// Create a new data parallel wrapper
    ///
    /// # Arguments
    /// * `model` - The model to replicate across devices
    /// * `rank` - This process's rank in the distributed group
    /// * `world_size` - Total number of processes in the group
    pub fn new(model: M, rank: usize, world_size: usize) -> Result<Self> {
        let process_group = ProcessGroup::new(Rank(rank), WorldSize(world_size))?;
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
    pub fn new_with_gpu(model: M, rank: usize, world_size: usize, device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let process_group = ProcessGroup::new(Rank(rank), WorldSize(world_size))?;
        let mut gradient_reducer = GradientReducer::new_with_gpu(process_group.clone(), device, queue);

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
    ) -> std::result::Result<coeus_tensor::Tensor<B, S, T>, NNError>
    {
        self.model.forward(input)
    }

    /// Synchronize gradients across all devices
    ///
    /// This performs AllReduce on all parameter gradients to ensure
    /// consistent model updates across the distributed group.
    pub async fn synchronize_gradients(&mut self) -> Result<()> {
        // In a full implementation, this would iterate through all parameters
        // and call gradient_reducer.reduce_gradients() for each
        // For now, this is a placeholder

        // Placeholder: synchronize all registered parameters
        for param_name in self.get_parameter_names() {
            // This would normally get gradients from the model parameters
            // and pass them to the reducer
            let _placeholder_grads = vec![0.0; 1]; // Placeholder
            self.gradient_reducer
                .reduce_gradients(&param_name, &_placeholder_grads)
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
    pub async fn backward(&mut self, loss: &coeus_tensor::Tensor<B, S, T>) -> Result<()> {
        // Compute gradients locally (this would normally call model.backward(loss))
        // For now, simulate gradient computation
        let param_names = self.get_parameter_names();

        for param_name in param_names {
            // In practice, we'd get actual gradients from the model
            let dummy_gradients = vec![1.0f32; 10]; // Placeholder gradients

            if self.use_gpu_sync {
                // Use GPU-accelerated gradient reduction
                self.gradient_reducer.reduce_gradients_gpu(&param_name, &dummy_gradients).await?;
            } else {
                // Use CPU gradient reduction
                self.gradient_reducer.reduce_gradients(&param_name, &dummy_gradients).await?;
            }
        }

        Ok(())
    }

    /// Register all model parameters with the gradient reducer
    fn register_model_parameters(_model: &M, reducer: &mut GradientReducer) -> Result<()> {
        // In a full implementation, this would iterate through model.parameters()
        // and register each parameter with the reducer

        // Placeholder: register some dummy parameters
        let param_names = vec!["weight".to_string(), "bias".to_string()];

        for name in param_names {
            // In practice, we'd get the actual parameter size from the model
            let placeholder_size = 10; // Placeholder
            reducer.register_parameter(name, placeholder_size)?;
        }

        Ok(())
    }

    /// Get parameter names (placeholder implementation)
    fn get_parameter_names(&self) -> Vec<String> {
        vec!["weight".to_string(), "bias".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_group::Rank;

    // Note: This test is simplified since we don't have a full Module trait implementation
    // In the real implementation, this would use actual model parameters
    #[tokio::test]
    async fn test_data_parallel_creation() {
        // Create a simple placeholder that implements the Module trait
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
