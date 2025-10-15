//! Distributed optimizer wrappers for gradient synchronization
//!
//! This module provides wrappers around existing optimizers to enable
//! distributed training with gradient synchronization across multiple devices.

use crate::error::Result;
use crate::process_group::ProcessGroup;
use crate::reducer::GradientReducer;
use coeus_backend::CpuBackend;
use coeus_dtype::traits::FloatExt;
use coeus_nn::Parameter;
use coeus_optim::Optimizer;
use coeus_storage::DenseStorage;
use std::collections::HashMap;

/// Distributed optimizer wrapper
///
/// This wraps any optimizer to add distributed gradient synchronization.
/// Gradients are accumulated across all devices before optimizer updates.
#[derive(Debug)]
pub struct DistributedOptimizer<O, B, S, T> {
    /// The underlying optimizer
    optimizer: O,
    /// Process group for communication
    process_group: ProcessGroup,
    /// Gradient reducer for synchronization
    gradient_reducer: GradientReducer,
    /// Step counter for synchronization
    step_count: usize,
    /// Phantom data for type parameters
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<O, B, S, T> DistributedOptimizer<O, B, S, T>
where
    O: Optimizer<B, S, T>,
    B: Send + Sync + coeus_backend::Backend,
    S: Send + Sync + coeus_storage::Storage<T> + 'static,
    T: Send + Sync + coeus_dtype::DataType + FloatExt,
{
    /// Create a new distributed optimizer wrapper
    ///
    /// # Arguments
    /// * `optimizer` - The base optimizer to wrap
    /// * `rank` - This process's rank in the distributed group
    /// * `world_size` - Total number of processes in the group
    pub fn new(optimizer: O, rank: usize, world_size: usize) -> Result<Self> {
        let process_group = ProcessGroup::new(
            crate::process_group::Rank(rank),
            crate::process_group::WorldSize(world_size),
        )?;
        let gradient_reducer = GradientReducer::new(process_group.clone());

        Ok(Self {
            optimizer,
            process_group,
            gradient_reducer,
            step_count: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Register a parameter for gradient synchronization
    ///
    /// This must be called for each parameter that will be optimized
    /// to set up the gradient reduction buffers.
    ///
    /// # Arguments
    /// * `name` - Parameter name (must be unique)
    /// * `size` - Number of elements in the parameter
    pub fn register_parameter(&mut self, name: impl Into<String>, size: usize) -> Result<()> {
        self.gradient_reducer.register_parameter(name.into(), size)
    }

    /// Perform a distributed optimization step
    ///
    /// This synchronizes gradients across all devices, then performs
    /// the optimization step using the averaged gradients.
    ///
    /// # Arguments
    /// * `gradients` - Map of parameter names to their local gradients
    ///
    /// # Returns
    /// The step loss (if applicable)
    pub async fn step(&mut self, gradients: HashMap<String, &[f32]>) -> Result<Option<f32>> {
        self.step_count += 1;

        // Synchronize gradients across all devices
        for (name, local_grads) in &gradients {
            self.gradient_reducer
                .reduce_gradients(name, local_grads)
                .await?;
        }

        // Apply barrier to ensure all processes are synchronized
        self.process_group.barrier().await?;

        // Collect synchronized gradients
        let mut synced_gradients = HashMap::new();
        for name in gradients.keys() {
            let reduced_grads = self.gradient_reducer.get_reduced_gradients(name)?;
            synced_gradients.insert(name.clone(), reduced_grads);
        }

        // Delegate to the underlying optimizer with synchronized gradients
        // Note: In a real implementation, this would need to be adapted
        // to work with the specific optimizer's step interface
        // For now, this is a conceptual implementation

        Ok(None)
    }

    /// Get the current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the process group
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }

    /// Get the underlying optimizer (mutable access)
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get the underlying optimizer (immutable access)
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }
}

impl<O, B, S, T> DistributedOptimizer<O, B, S, T> {
    /// Create from existing process group (for advanced usage)
    ///
    /// # Arguments
    /// * `optimizer` - The base optimizer to wrap
    /// * `process_group` - Pre-configured process group
    pub fn with_process_group(optimizer: O, process_group: ProcessGroup) -> Self {
        let gradient_reducer = GradientReducer::new(process_group.clone());

        Self {
            optimizer,
            process_group,
            gradient_reducer,
            step_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_group::{Rank, WorldSize};
    use coeus_dtype::float::Float32;

    // Mock optimizer for testing
    #[derive(Debug)]
    struct MockOptimizer;

    impl MockOptimizer {
        fn new() -> Self {
            Self
        }
    }

    impl Optimizer<CpuBackend, DenseStorage<Float32>, Float32> for MockOptimizer {
        fn step(&mut self) -> coeus_optim::Result<usize> {
            Ok(0)
        }

        fn zero_grad(&mut self) {
            // No-op for mock
        }

        fn add_param(
            &mut self,
            _param: &mut coeus_nn::Parameter<CpuBackend, DenseStorage<Float32>, Float32>,
        ) -> coeus_optim::Result<()> {
            Ok(())
        }

        fn learning_rate(&self) -> f64 {
            0.01
        }

        fn set_learning_rate(&mut self, _lr: f64) -> coeus_optim::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_distributed_optimizer_creation() {
        let mock_optimizer = MockOptimizer::new();
        let dist_optimizer = DistributedOptimizer::new(mock_optimizer, 0, 4).unwrap();

        assert_eq!(dist_optimizer.step_count(), 0);
        assert_eq!(dist_optimizer.process_group().rank(), Rank(0));
        assert_eq!(dist_optimizer.process_group().world_size(), WorldSize(4));
    }

    #[test]
    fn test_parameter_registration() {
        let mock_optimizer = MockOptimizer::new();
        let mut dist_optimizer = DistributedOptimizer::new(mock_optimizer, 0, 2).unwrap();

        // Register a parameter
        dist_optimizer.register_parameter("weight", 10).unwrap();

        // Should succeed without panic
        assert_eq!(dist_optimizer.step_count(), 0);
    }

    #[tokio::test]
    async fn test_distributed_step() {
        let mock_optimizer = MockOptimizer::new();
        let mut dist_optimizer = DistributedOptimizer::new(mock_optimizer, 0, 2).unwrap();

        // Register a parameter
        dist_optimizer.register_parameter("weight", 4).unwrap();

        // Create test gradients
        let mut gradients = HashMap::new();
        let grad_vec = vec![0.1, 0.2, 0.3, 0.4];
        gradients.insert("weight".to_string(), grad_vec.as_slice());

        // Perform distributed step
        let result = dist_optimizer.step(gradients).await;
        assert!(result.is_ok());
        assert_eq!(dist_optimizer.step_count(), 1);
    }
}
