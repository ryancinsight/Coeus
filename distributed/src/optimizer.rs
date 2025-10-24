//! Distributed optimizer wrappers for gradient synchronization
//!
//! This module provides wrappers around existing optimizers to enable
//! distributed training with gradient synchronization across multiple devices.

use crate::error::Result;
use crate::process_group::ProcessGroup;
use crate::reducer::GradientReducer;
use coeus_dtype::float::Float32;
use coeus_optim::Optimizer;
use coeus_tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

/// Distributed optimizer wrapper
///
/// This wraps any optimizer to add distributed gradient synchronization.
/// Gradients are accumulated across all devices before optimizer updates.
/// Currently constrained to Float32 dtype for distributed training compatibility.
#[derive(Debug)]
pub struct DistributedOptimizer<O, B, S> {
    /// The underlying optimizer
    optimizer: O,
    /// Process group for communication
    process_group: Arc<ProcessGroup>,
    /// Gradient reducer for synchronization
    gradient_reducer: GradientReducer,
    /// Step counter for synchronization
    step_count: usize,
    /// Phantom data for type parameters
    _phantom: std::marker::PhantomData<(B, S)>,
}

impl<O, B, S> DistributedOptimizer<O, B, S>
where
    O: Optimizer<B, S, Float32>,
    B: Send + Sync + coeus_backend::Backend,
    S: Send
        + Sync
        + coeus_storage::Storage<Float32>
        + Clone
        + coeus_storage::StorageFromVec<Float32>
        + 'static,
{
    /// Create a new distributed optimizer wrapper
    ///
    /// # Arguments
    /// * `optimizer` - The base optimizer to wrap
    /// * `rank` - This process's rank in the distributed group
    /// * `world_size` - Total number of processes in the group
    pub fn new(optimizer: O, rank: usize, world_size: usize) -> Result<Self> {
        let process_group = Arc::new(ProcessGroup::new(
            crate::process_group::Rank(rank),
            crate::process_group::WorldSize(world_size),
        )?);
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

        // Apply synchronized gradients to the underlying optimizer
        // For each parameter in the optimizer, set its gradient to the synchronized version
        let optimizer_params = self.optimizer.parameters();
        for (i, parameter) in optimizer_params.into_iter().enumerate() {
            if let Some(grad_data) = synced_gradients.get(&format!("param_{}", i)) {
                // Convert the synchronized gradient data back to tensor format
                let float32_data: Vec<Float32> =
                    grad_data.iter().map(|&x| Float32::new(x)).collect();
                // Create gradient tensor using the same backend as the parameter
                let grad_tensor = Tensor::from_vec_with_backend(
                    float32_data,
                    parameter.shape().dims(),
                    parameter.backend().clone(),
                )?;
                parameter.set_grad(grad_tensor)?;
            }
        }

        // Perform the optimization step with synchronized gradients
        self.optimizer.step()?;

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

impl<O, B, S> DistributedOptimizer<O, B, S>
where
    O: Optimizer<B, S, Float32>,
    B: Send + Sync + coeus_backend::Backend,
    S: Send
        + Sync
        + coeus_storage::Storage<Float32>
        + Clone
        + coeus_storage::StorageFromVec<Float32>
        + 'static,
{
    /// Create from existing process group (for advanced usage)
    ///
    /// # Arguments
    /// * `optimizer` - The base optimizer to wrap
    /// * `process_group` - Pre-configured process group
    pub fn with_process_group(optimizer: O, process_group: Arc<ProcessGroup>) -> Self {
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
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;

    // Mock optimizer for testing
    #[derive(Debug)]
    struct MockOptimizer;

    impl MockOptimizer {
        fn new() -> Self {
            Self
        }
    }

    impl Optimizer<CpuBackend<Float32>, DenseStorage<Float32>, Float32> for MockOptimizer {
        fn name(&self) -> &str {
            "MockOptimizer"
        }

        fn parameters(&self) -> Vec<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
            vec![]
        }

        fn named_parameters(
            &self,
        ) -> std::collections::HashMap<
            String,
            Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        > {
            std::collections::HashMap::new()
        }

        fn add_param(
            &mut self,
            _param: &mut Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
            _name: String,
        ) -> std::result::Result<(), coeus_optim::error::OptimError> {
            // No-op for mock
            Ok(())
        }

        fn remove_param(&mut self, _name: &str) {
            // No-op for mock
        }

        fn has_param(&self, _name: &str) -> bool {
            false
        }

        fn lr(&self) -> f64 {
            0.01
        }

        fn set_lr(&mut self, _lr: f64) -> std::result::Result<(), coeus_optim::error::OptimError> {
            // No-op for mock
            Ok(())
        }

        fn weight_decay(&self) -> f64 {
            0.0
        }

        fn set_weight_decay(
            &mut self,
            _weight_decay: f64,
        ) -> std::result::Result<(), coeus_optim::error::OptimError> {
            // No-op for mock
            Ok(())
        }

        fn zero_grad(&mut self) {
            // No-op for mock
        }

        fn step(&mut self) -> std::result::Result<usize, coeus_optim::error::OptimError> {
            Ok(1) // Mock step count
        }

        fn state_dict(
            &self,
        ) -> std::collections::HashMap<
            String,
            Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        > {
            std::collections::HashMap::new()
        }

        fn load_state_dict(
            &mut self,
            _state_dict: std::collections::HashMap<
                String,
                Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
            >,
        ) -> std::result::Result<(), coeus_optim::error::OptimError> {
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
