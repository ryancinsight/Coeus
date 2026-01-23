//! Distributed Training Support
//!
//! Simplified distributed training simulation for demonstrating
//! multi-device training concepts.

#[cfg(not(feature = "autograd"))]
use crate::autograd_stub::backward;
use crate::core::error::{NNError, Result};
use crate::core::module::Module;
#[cfg(feature = "autograd")]
use autograd::backward;
use backend::Backend;
use dtype::DataType;
// use std::collections::HashMap;
use storage::DenseStorage;
use tensor::Tensor;

/// Simplified distributed training simulator
///
/// This simulates distributed training across multiple devices for demonstration
/// purposes. In a real implementation, this would use actual network communication
/// and hardware acceleration.
#[derive(Debug)]
pub struct Distributed<M, B, T> {
    /// The wrapped neural network model
    model: M,
    /// Number of simulated processes/devices
    world_size: usize,
    /// Current process rank (0-based)
    rank: usize,
    /// Phantom data for type system
    _phantom: std::marker::PhantomData<(B, T)>,
}

impl<M, B, T> Distributed<M, B, T>
where
    M: Module<B, DenseStorage<T>, T> + Send + Sync,
    B: Backend<Data = T> + Send + Sync + Default + Clone,
    T: DataType + dtype::traits::FloatExt + Send + Sync,
{
    /// Create a new distributed training simulator
    ///
    /// # Arguments
    /// * `model` - The neural network model to distribute
    /// * `rank` - This process's rank in the distributed group (0-based)
    /// * `world_size` - Total number of processes in the distributed group
    ///
    /// # Returns
    /// A distributed training simulator ready for demonstration
    pub fn new(model: M, rank: usize, world_size: usize) -> Result<Self> {
        if rank >= world_size {
            return Err(NNError::InvalidConfiguration {
                message: "Process rank must be less than world size".to_string(),
            });
        }

        Ok(Self {
            model,
            world_size,
            rank,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Forward pass through the distributed model
    ///
    /// # Arguments
    /// * `input` - Input tensor batch
    ///
    /// # Returns
    /// Model output tensor
    pub fn forward(
        &self,
        input: &Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        self.model.forward(input)
    }

    /// Simulated backward pass with gradient synchronization
    ///
    /// In a real distributed system, this would synchronize gradients across processes.
    /// Here, it just performs local backpropagation.
    ///
    /// # Arguments
    /// * `loss` - Loss tensor to backpropagate from
    pub fn backward(&mut self, loss: &Tensor<B, DenseStorage<T>, T>) -> Result<()> {
        // In a real distributed system, this would:
        // 1. Perform local backward pass
        // 2. Synchronize gradients across all processes using AllReduce
        // 3. Average gradients across processes

        // For simulation, just perform local backward
        backward(loss, None, false, false).map_err(|e| NNError::TrainingError {
            message: format!("Backward pass failed: {}", e),
        })
    }

    /// Get the underlying model (useful for saving checkpoints)
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable access to the underlying model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get the current process rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size (total number of processes)
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this is the master process (rank 0)
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Simulated barrier synchronization
    ///
    /// In a real distributed system, this would synchronize all processes.
    pub fn barrier(&self) -> Result<()> {
        // Simulate barrier by doing nothing (in real distributed training,
        // this would block until all processes reach this point)
        Ok(())
    }

    /// Get training statistics for monitoring distributed training
    pub fn training_stats(&self) -> DistributedStats {
        DistributedStats {
            rank: self.rank,
            world_size: self.world_size,
            gradient_sync_count: 0,    // Simulated
            communication_overhead: 0, // Simulated
        }
    }
}

/// Statistics for monitoring distributed training performance
#[derive(Debug, Clone)]
pub struct DistributedStats {
    /// Current process rank
    pub rank: usize,
    /// Total number of processes
    pub world_size: usize,
    /// Number of gradient synchronizations performed
    pub gradient_sync_count: u64,
    /// Total bytes communicated for gradient synchronization
    pub communication_overhead: u64,
}

impl DistributedStats {
    /// Calculate communication efficiency (lower is better)
    pub fn communication_efficiency(&self) -> f64 {
        if self.gradient_sync_count == 0 {
            0.0
        } else {
            self.communication_overhead as f64 / self.gradient_sync_count as f64
        }
    }

    /// Get parallelization factor
    pub fn parallelization_factor(&self) -> usize {
        self.world_size
    }
}

// Type aliases for common distributed configurations
/// CPU-based distributed training - NOTE: CpuBackend now requires generic parameter
// pub type DistributedCpu<M, T> = Distributed<M, backend::CpuBackend, T>;

/// GPU-based distributed training
#[cfg(feature = "gpu")]
pub type DistributedGpu<M, T> = Distributed<M, backend::GpuBackend<T>, T>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;
    use backend::CpuBackend;
    use dtype::float::Float32;

    #[test]
    fn test_distributed_creation() {
        let model = Linear::<CpuBackend<Float32>, _, Float32>::new(10, 5).unwrap();
        let distributed = Distributed::new(model, 0, 1).unwrap();

        assert_eq!(distributed.rank(), 0);
        assert_eq!(distributed.world_size(), 1);
        assert!(distributed.is_master());
    }

    #[test]
    fn test_distributed_forward() {
        let model = Linear::<CpuBackend<Float32>, _, Float32>::new(4, 2).unwrap();
        let distributed = Distributed::new(model, 0, 1).unwrap();

        let input = Tensor::<CpuBackend<Float32>, _, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[1, 4],
        )
        .unwrap();

        let output = distributed.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2]);
    }

    #[test]
    fn test_distributed_stats() {
        let model = Linear::<CpuBackend<Float32>, _, Float32>::new(4, 2).unwrap();
        let distributed = Distributed::new(model, 1, 4).unwrap();

        let stats = distributed.training_stats();
        assert_eq!(stats.rank, 1);
        assert_eq!(stats.world_size, 4);
        assert_eq!(stats.parallelization_factor(), 4);
    }

    #[test]
    fn test_sequential_distributed() {
        // TODO: Enable this test once Sequential implements Send + Sync
        // Currently Sequential stores Box<dyn Module> which is not Send+Sync by default
        // causing Distributed wrapper to fail trait bounds.

        /*
        let mut model = Sequential::new();
        model.add_module("linear1".to_string(), Linear::new(4, 3).unwrap());
        model.add_module("linear2".to_string(), Linear::new(3, 2).unwrap());

        let distributed = Distributed::new(model, 0, 2).unwrap();
        assert_eq!(distributed.world_size(), 2);
        */
    }
}
