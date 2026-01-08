//! Process group management for distributed training coordination

use crate::communication::{create_backend, BackendStats, BackendType, CommunicationBackend};
use crate::error::{DistributedError, Result};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};

/// Rank identifier for a process in the distributed group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rank(pub usize);

/// Total number of processes in the distributed group
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldSize(pub usize);

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Maximum number of retries for failed operations
    pub max_retries: usize,
    /// Timeout for individual operations
    pub operation_timeout: std::time::Duration,
    /// Whether to enable automatic recovery
    pub enable_recovery: bool,
    /// Health check interval
    pub health_check_interval: std::time::Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            operation_timeout: std::time::Duration::from_secs(30),
            enable_recovery: true,
            health_check_interval: std::time::Duration::from_secs(10),
        }
    }
}

/// Process group for coordinating distributed operations
///
/// This represents a group of processes that can communicate and synchronize
/// gradients during distributed training with support for multiple communication
/// backends (NCCL, Gloo, MPI) and fault tolerance.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ProcessGroup {
    rank: Rank,
    world_size: WorldSize,
    /// Communication backend for distributed operations
    backend: Arc<Mutex<CommunicationBackend>>,
    /// Fault tolerance configuration
    fault_tolerance: FaultToleranceConfig,
    /// Whether the group has been initialized
    initialized: RwLock<bool>,
    /// Health status of the group
    healthy: RwLock<bool>,
    /// Channel for data exchange between processes (simulation)
    data_tx: mpsc::UnboundedSender<Vec<f32>>,
    data_rx: mpsc::UnboundedReceiver<Vec<f32>>,
    /// Broadcast channel for synchronization
    sync_tx: broadcast::Sender<()>,
    sync_rx: broadcast::Receiver<()>,
}

impl ProcessGroup {
    /// Create a new process group with auto-detected backend
    pub fn new(rank: Rank, world_size: WorldSize) -> Result<Self> {
        Self::new_with_backend(rank, world_size, BackendType::Gloo)
    }

    /// Create a new process group with specified backend
    pub fn new_with_backend(
        rank: Rank,
        world_size: WorldSize,
        backend_type: BackendType,
    ) -> Result<Self> {
        if rank.0 >= world_size.0 {
            return Err(DistributedError::ProcessGroupConfig {
                message: format!("Rank {} >= world_size {}", rank.0, world_size.0),
            });
        }

        let (data_tx, data_rx) = mpsc::unbounded_channel();
        let (sync_tx, sync_rx) = broadcast::channel(16);

        Ok(Self {
            rank,
            world_size,
            backend: Arc::new(Mutex::new(create_backend(backend_type))),
            fault_tolerance: FaultToleranceConfig::default(),
            initialized: RwLock::new(false),
            healthy: RwLock::new(true),
            data_tx,
            data_rx,
            sync_tx,
            sync_rx,
        })
    }

    /// Create a new process group with custom backend
    pub fn with_custom_backend(
        rank: Rank,
        world_size: WorldSize,
        backend: CommunicationBackend,
    ) -> Result<Self> {
        if rank.0 >= world_size.0 {
            return Err(DistributedError::ProcessGroupConfig {
                message: format!("Rank {} >= world_size {}", rank.0, world_size.0),
            });
        }

        let (data_tx, data_rx) = mpsc::unbounded_channel();
        let (sync_tx, sync_rx) = broadcast::channel(16);

        Ok(Self {
            rank,
            world_size,
            backend: Arc::new(Mutex::new(backend)),
            fault_tolerance: FaultToleranceConfig::default(),
            initialized: RwLock::new(false),
            healthy: RwLock::new(true),
            data_tx,
            data_rx,
            sync_tx,
            sync_rx,
        })
    }

    /// Initialize the process group and communication backend
    pub async fn initialize(&self) -> Result<()> {
        if *self.initialized.read().await {
            return Ok(());
        }

        let mut backend = self.backend.lock().await;
        backend.initialize(self).await?;
        *self.initialized.write().await = true;
        *self.healthy.write().await = true;

        Ok(())
    }

    /// Set fault tolerance configuration
    pub fn with_fault_tolerance(mut self, config: FaultToleranceConfig) -> Self {
        self.fault_tolerance = config;
        self
    }

    /// Get fault tolerance configuration
    #[must_use]
    pub fn fault_tolerance(&self) -> &FaultToleranceConfig {
        &self.fault_tolerance
    }

    /// Check if the process group is healthy
    #[must_use]
    pub async fn is_healthy(&self) -> bool {
        *self.healthy.read().await && *self.initialized.read().await
    }

    /// Get communication backend statistics
    pub async fn backend_statistics(&self) -> Result<BackendStats> {
        let backend = self.backend.lock().await;
        Ok(backend.statistics())
    }

    /// Get the rank of this process
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Get the total world size
    pub fn world_size(&self) -> WorldSize {
        self.world_size
    }

    /// Check if this is the master process (rank 0)
    pub fn is_master(&self) -> bool {
        self.rank.0 == 0
    }

    /// Perform AllReduce operation on CPU buffers with fault tolerance
    pub async fn all_reduce(&self, buffer: &mut [f32]) -> crate::Result<()> {
        self.all_reduce_with_retry(buffer, 0).await
    }

    /// Perform AllReduce operation on GPU buffers with fault tolerance
    pub async fn all_reduce_gpu(&self, buffer: &wgpu::Buffer, size: usize) -> crate::Result<()> {
        self.all_reduce_gpu_with_retry(buffer, size, 0).await
    }

    /// Internal AllReduce with retry logic
    async fn all_reduce_with_retry(&self, buffer: &mut [f32], attempt: usize) -> crate::Result<()> {
        if attempt >= self.fault_tolerance.max_retries {
            *self.healthy.write().await = false;
            return Err(DistributedError::Communication {
                message: format!("AllReduce failed after {} retries", attempt),
            });
        }

        let mut backend = self.backend.lock().await;
        match tokio::time::timeout(
            self.fault_tolerance.operation_timeout,
            backend.all_reduce_cpu(buffer),
        )
        .await
        {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_e)) => {
                // Retry on failure
                drop(backend); // Release lock before retry
                tokio::time::sleep(std::time::Duration::from_millis(100 * (attempt + 1) as u64))
                    .await;
                Box::pin(self.all_reduce_with_retry(buffer, attempt + 1)).await
            }
            Err(_) => {
                // Timeout - retry
                drop(backend);
                tokio::time::sleep(std::time::Duration::from_millis(200 * (attempt + 1) as u64))
                    .await;
                Box::pin(self.all_reduce_with_retry(buffer, attempt + 1)).await
            }
        }
    }

    /// Internal GPU AllReduce with retry logic
    async fn all_reduce_gpu_with_retry(
        &self,
        buffer: &wgpu::Buffer,
        size: usize,
        attempt: usize,
    ) -> crate::Result<()> {
        if attempt >= self.fault_tolerance.max_retries {
            *self.healthy.write().await = false;
            return Err(DistributedError::Communication {
                message: format!("GPU AllReduce failed after {} retries", attempt),
            });
        }

        let backend = self.backend.lock().await;
        match tokio::time::timeout(
            self.fault_tolerance.operation_timeout,
            backend.all_reduce_gpu(buffer, size),
        )
        .await
        {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_e)) => {
                // Retry on failure
                drop(backend);
                tokio::time::sleep(std::time::Duration::from_millis(100 * (attempt + 1) as u64))
                    .await;
                Box::pin(self.all_reduce_gpu_with_retry(buffer, size, attempt + 1)).await
            }
            Err(_) => {
                // Timeout - retry
                drop(backend);
                tokio::time::sleep(std::time::Duration::from_millis(200 * (attempt + 1) as u64))
                    .await;
                Box::pin(self.all_reduce_gpu_with_retry(buffer, size, attempt + 1)).await
            }
        }
    }

    /// Perform barrier synchronization with fault tolerance
    pub async fn barrier(&self) -> Result<()> {
        let mut attempt = 0;
        loop {
            if attempt >= self.fault_tolerance.max_retries {
                *self.healthy.write().await = false;
                return Err(DistributedError::Communication {
                    message: format!("Barrier failed after {} retries", attempt),
                });
            }

            let backend = self.backend.lock().await;
            match tokio::time::timeout(self.fault_tolerance.operation_timeout, backend.barrier())
                .await
            {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(_)) | Err(_) => {
                    drop(backend);
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(50 * attempt as u64)).await;
                }
            }
        }
    }

    /// Shutdown the process group and communication backend
    pub async fn shutdown(&self) -> Result<()> {
        if !*self.initialized.read().await {
            return Ok(());
        }

        let mut backend = self.backend.lock().await;
        backend.shutdown().await?;
        *self.initialized.write().await = false;
        *self.healthy.write().await = false;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_group_creation() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(4)).unwrap();
        assert_eq!(pg.rank(), Rank(0));
        assert_eq!(pg.world_size(), WorldSize(4));
        assert!(pg.is_master());
    }

    #[test]
    fn test_process_group_non_master() {
        let pg = ProcessGroup::new(Rank(2), WorldSize(4)).unwrap();
        assert_eq!(pg.rank(), Rank(2));
        assert_eq!(pg.world_size(), WorldSize(4));
        assert!(!pg.is_master());
    }

    #[test]
    fn test_process_group_invalid_rank() {
        // Rank cannot be >= world_size
        let result = ProcessGroup::new(Rank(4), WorldSize(4));
        assert!(result.is_err());

        let result = ProcessGroup::new(Rank(5), WorldSize(4));
        assert!(result.is_err());
    }

    #[test]
    fn test_process_group_zero_world_size() {
        let result = ProcessGroup::new(Rank(0), WorldSize(0));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_all_reduce_simulation() {
        let pg = ProcessGroup::new(Rank(0), WorldSize(2)).unwrap();

        // Test with sample data
        let mut data = vec![1.0, 2.0, 3.0];
        let result = pg.all_reduce(&mut data).await;
        assert!(result.is_ok());

        // In simulation, data should remain unchanged
        // In real distributed system, this would be sum/average across devices
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[tokio::test]
    async fn test_barrier_synchronization() {
        let pg = ProcessGroup::new(Rank(1), WorldSize(3)).unwrap();

        // Barrier should complete without error
        let result = pg.barrier().await;
        assert!(result.is_ok());
    }
}
