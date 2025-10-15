//! # Distributed Storage Abstractions
//!
//! Provides storage types and operations for distributed tensors across multiple devices.
//! Supports tensor parallelism, data parallelism, and multi-device operations.
//!
//! ## Architecture
//!
//! Distributed storage enables:
//! - **Tensor Parallelism**: Splitting large tensors across devices
//! - **Data Parallelism**: Replicating models with distributed data
//! - **Pipeline Parallelism**: Model stages across devices
//! - **Zero Redundancy**: Memory-efficient distributed training
//!
//! ## Sharding Strategies
//!
//! - **Row-wise Sharding**: Split tensors along the first dimension
//! - **Column-wise Sharding**: Split tensors along the last dimension
//! - **Block-wise Sharding**: 2D block decomposition for matrix operations
//! - **ZeRO Sharding**: Optimizer state partitioning for memory efficiency

use crate::{AsAny, DataType, Result, Shape, Storage, StorageError};
use alloc::{boxed::Box, vec, vec::Vec};
use core::fmt;

/// Device identifier for distributed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

/// Sharding strategy for distributed tensors
#[derive(Debug, Clone, PartialEq)]
pub enum ShardingStrategy {
    /// No sharding - replicate on all devices
    Replicated,
    /// Split along first dimension (rows)
    RowWise,
    /// Split along last dimension (columns)
    ColumnWise,
    /// 2D block decomposition
    BlockWise { block_rows: usize, block_cols: usize },
    /// ZeRO-style sharding for optimizer states
    ZeroSharding,
}

/// Distributed shard information
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Device this shard resides on
    pub device_id: DeviceId,
    /// Offset within the global tensor
    pub offset: Vec<usize>,
    /// Shape of this shard
    pub shape: Shape,
    /// Index of this shard in the global tensor
    pub shard_index: usize,
}

/// Enum representing different storage types for distributed tensors
#[derive(Debug, Clone)]
pub enum StorageVariant<T: DataType> {
    /// Dense storage
    Dense(crate::dense::DenseStorage<T>),
    // Sparse storage variants would be added here
}

impl<T: DataType> StorageVariant<T> {
    /// Get the underlying storage as a trait object reference
    #[must_use]
    pub fn as_storage(&self) -> &dyn Storage<T> {
        match self {
            Self::Dense(dense) => dense,
        }
    }
}

/// Distributed storage across multiple devices
///
/// Manages tensor shards distributed across different devices with
/// automatic communication and synchronization.
#[derive(Debug, Clone)]
pub struct DistributedStorage<T: DataType> {
    /// Global shape of the distributed tensor
    global_shape: Shape,
    /// Sharding strategy used
    sharding_strategy: ShardingStrategy,
    /// Information about each shard
    shards: Vec<ShardInfo>,
    /// Device assignments for each shard
    device_assignments: Vec<DeviceId>,
    /// Local storage for shards on this device (if any)
    local_shards: Vec<(usize, StorageVariant<T>)>,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: DataType> DistributedStorage<T> {
    /// Create a new distributed storage with specified sharding
    ///
    /// # Arguments
    /// * `global_shape` - Shape of the full tensor
    /// * `sharding_strategy` - How to shard the tensor
    /// * `device_ids` - Available device IDs for distribution
    ///
    /// # Errors
    ///
    /// Returns error if sharding configuration is invalid.
    pub fn new(
        global_shape: &[usize],
        sharding_strategy: ShardingStrategy,
        device_ids: &[DeviceId],
    ) -> Result<Self> {
        let global_shape = Shape::new(global_shape)?;
        if device_ids.is_empty() {
            return Err(StorageError::InvalidShape {
                reason: "must provide at least one device",
            });
        }

        let shards = Self::compute_shards(&global_shape, &sharding_strategy, device_ids)?;
        let device_assignments = (0..shards.len())
            .map(|i| device_ids[i % device_ids.len()])
            .collect();

        Ok(Self {
            global_shape,
            sharding_strategy,
            shards,
            device_assignments,
            local_shards: Vec::new(),
            _phantom: core::marker::PhantomData,
        })
    }

    /// Compute shard information based on sharding strategy
    fn compute_shards(
        global_shape: &Shape,
        strategy: &ShardingStrategy,
        device_ids: &[DeviceId],
    ) -> Result<Vec<ShardInfo>> {
        match strategy {
            ShardingStrategy::Replicated => {
                // Create one shard per device, all with full shape
                let shards = device_ids
                    .iter()
                    .enumerate()
                    .map(|(i, &device_id)| ShardInfo {
                        device_id,
                        offset: vec![0; global_shape.ndim()],
                        shape: global_shape.clone(),
                        shard_index: i,
                    })
                    .collect();
                Ok(shards)
            }
            ShardingStrategy::RowWise => {
                Self::compute_row_wise_shards(global_shape, device_ids)
            }
            ShardingStrategy::ColumnWise => {
                Self::compute_column_wise_shards(global_shape, device_ids)
            }
            ShardingStrategy::BlockWise { block_rows, block_cols } => {
                Self::compute_block_wise_shards(global_shape, *block_rows, *block_cols, device_ids)
            }
            ShardingStrategy::ZeroSharding => {
                // ZeRO sharding distributes optimizer states
                Self::compute_zero_shards(global_shape, device_ids)
            }
        }
    }

    /// Compute row-wise shards (split along first dimension)
    fn compute_row_wise_shards(
        global_shape: &Shape,
        device_ids: &[DeviceId],
    ) -> Result<Vec<ShardInfo>> {
        let dims = global_shape.dims();
        if dims.is_empty() {
            return Err(StorageError::InvalidShape {
                reason: "cannot shard scalar tensors",
            });
        }

        let total_rows = dims[0];
        let num_devices = device_ids.len();
        let base_rows_per_shard = total_rows / num_devices;
        let extra_rows = total_rows % num_devices;

        let mut shards = Vec::new();
        let mut current_offset = 0;

        for (i, &device_id) in device_ids.iter().enumerate() {
            let rows_in_shard = base_rows_per_shard + if i < extra_rows { 1 } else { 0 };
            if rows_in_shard == 0 {
                continue; // Skip devices with no rows
            }

            let mut shard_dims = dims.to_vec();
            shard_dims[0] = rows_in_shard;

            shards.push(ShardInfo {
                device_id,
                offset: vec![current_offset],
                shape: Shape::new(&shard_dims)?,
                shard_index: i,
            });

            current_offset += rows_in_shard;
        }

        Ok(shards)
    }

    /// Compute column-wise shards (split along last dimension)
    fn compute_column_wise_shards(
        global_shape: &Shape,
        device_ids: &[DeviceId],
    ) -> Result<Vec<ShardInfo>> {
        let dims = global_shape.dims();
        if dims.len() < 2 {
            return Err(StorageError::InvalidShape {
                reason: "column-wise sharding requires at least 2D tensors",
            });
        }

        let total_cols = *dims.last().unwrap();
        let num_devices = device_ids.len();
        let base_cols_per_shard = total_cols / num_devices;
        let extra_cols = total_cols % num_devices;

        let mut shards = Vec::new();
        let mut current_offset = 0;

        for (i, &device_id) in device_ids.iter().enumerate() {
            let cols_in_shard = base_cols_per_shard + if i < extra_cols { 1 } else { 0 };
            if cols_in_shard == 0 {
                continue;
            }

            let mut shard_dims = dims.to_vec();
            *shard_dims.last_mut().unwrap() = cols_in_shard;

            let mut offset = vec![0; dims.len()];
            *offset.last_mut().unwrap() = current_offset;

            shards.push(ShardInfo {
                device_id,
                offset,
                shape: Shape::new(&shard_dims)?,
                shard_index: i,
            });

            current_offset += cols_in_shard;
        }

        Ok(shards)
    }

    /// Compute block-wise shards (2D decomposition)
    fn compute_block_wise_shards(
        global_shape: &Shape,
        block_rows: usize,
        block_cols: usize,
        device_ids: &[DeviceId],
    ) -> Result<Vec<ShardInfo>> {
        let dims = global_shape.dims();
        if dims.len() != 2 {
            return Err(StorageError::InvalidShape {
                reason: "block-wise sharding requires 2D tensors",
            });
        }

        let (rows, cols) = (dims[0], dims[1]);
        let blocks_per_row = (rows + block_rows - 1) / block_rows;
        let blocks_per_col = (cols + block_cols - 1) / block_cols;
        let total_blocks = blocks_per_row * blocks_per_col;

        if device_ids.len() > total_blocks {
            return Err(StorageError::InvalidShape {
                reason: "more devices than blocks in block-wise sharding",
            });
        }

        let mut shards = Vec::new();

        for (i, &device_id) in device_ids.iter().enumerate() {
            let block_row_idx = i / blocks_per_col;
            let block_col_idx = i % blocks_per_col;

            let start_row = block_row_idx * block_rows;
            let start_col = block_col_idx * block_cols;

            let actual_rows = core::cmp::min(block_rows, rows.saturating_sub(start_row));
            let actual_cols = core::cmp::min(block_cols, cols.saturating_sub(start_col));

            if actual_rows == 0 || actual_cols == 0 {
                continue;
            }

            shards.push(ShardInfo {
                device_id,
                offset: vec![start_row, start_col],
                shape: Shape::new(&[actual_rows, actual_cols])?,
                shard_index: i,
            });
        }

        Ok(shards)
    }

    /// Compute ZeRO shards for optimizer state partitioning
    fn compute_zero_shards(
        global_shape: &Shape,
        device_ids: &[DeviceId],
    ) -> Result<Vec<ShardInfo>> {
        // ZeRO sharding distributes along the first dimension
        // but with additional logic for optimizer states
        Self::compute_row_wise_shards(global_shape, device_ids)
    }

    /// Get the global shape of the distributed tensor
    #[must_use]
    pub const fn global_shape(&self) -> &Shape {
        &self.global_shape
    }

    /// Get the sharding strategy
    #[must_use]
    pub const fn sharding_strategy(&self) -> &ShardingStrategy {
        &self.sharding_strategy
    }

    /// Get information about all shards
    #[must_use]
    pub fn shards(&self) -> &[ShardInfo] {
        &self.shards
    }

    /// Get device assignments for shards
    #[must_use]
    pub fn device_assignments(&self) -> &[DeviceId] {
        &self.device_assignments
    }

    /// Check if this device has any local shards
    #[must_use]
    pub fn has_local_shards(&self, device_id: DeviceId) -> bool {
        self.shards
            .iter()
            .any(|shard| shard.device_id == device_id)
    }

    /// Get shards assigned to a specific device
    #[must_use]
    pub fn get_device_shards(&self, device_id: DeviceId) -> Vec<&ShardInfo> {
        self.shards
            .iter()
            .filter(|shard| shard.device_id == device_id)
            .collect()
    }

    /// Add a local shard to this storage
    ///
    /// # Arguments
    /// * `shard_index` - Index of the shard in the global shard list
    /// * `storage` - The actual storage containing the shard data
    pub fn add_local_shard(&mut self, shard_index: usize, storage: StorageVariant<T>) -> Result<()> {
        if shard_index >= self.shards.len() {
            return Err(StorageError::IndexOutOfBounds {
                index: shard_index,
                bound: self.shards.len(),
            });
        }

        // Verify storage shape matches shard shape
        let storage_shape = match &storage {
            StorageVariant::Dense(dense) => dense.shape(),
        };
        if storage_shape != &self.shards[shard_index].shape {
            return Err(StorageError::ShapeMismatch {
                expected: self.shards[shard_index].shape.size(),
                actual: storage_shape.size(),
            });
        }

        self.local_shards.push((shard_index, storage));
        Ok(())
    }

    /// Get local shard by index
    #[must_use]
    pub fn get_local_shard(&self, shard_index: usize) -> Option<&StorageVariant<T>> {
        self.local_shards
            .iter()
            .find(|(idx, _)| *idx == shard_index)
            .map(|(_, storage)| storage)
    }

    /// Get all local shards
    #[must_use]
    pub fn local_shards(&self) -> &[(usize, StorageVariant<T>)] {
        &self.local_shards
    }

    /// Gather all shards into a complete tensor (expensive operation)
    ///
    /// This requires communication across all devices and should be used sparingly.
    /// In practice, this would involve MPI, NCCL, or similar communication libraries.
    pub async fn gather(&self) -> Result<Vec<T>> {
        // Placeholder implementation - in practice this would:
        // 1. Communicate with all devices to gather their shards
        // 2. Reconstruct the global tensor from shards
        // 3. Return the complete tensor data

        let total_elements = self.global_shape.size();
        let result = vec![T::default(); total_elements];

        // For each local shard, copy its data into the appropriate position
        for (shard_index, storage) in &self.local_shards {
            let shard_info = &self.shards[*shard_index];
            // This is a simplified copy - real implementation would handle offsets properly
            let shard_data = storage.as_storage().as_slice();
            // Copy logic would go here...
            let _ = shard_data; // Suppress unused variable warning
            let _ = shard_info; // Suppress unused variable warning
        }

        // Placeholder: return zeros
        Ok(result)
    }

    /// Scatter tensor data to appropriate shards
    ///
    /// Distributes data from a complete tensor to the appropriate device shards.
    pub async fn scatter(&self, _data: &[T]) -> Result<()> {
        // Placeholder implementation - in practice this would:
        // 1. Split the input data according to shard boundaries
        // 2. Send appropriate chunks to each device
        // 3. Create local shard storage on each device

        // For now, this is a no-op
        Ok(())
    }

    /// Perform AllReduce operation on distributed tensor
    ///
    /// Reduces values across all devices using the specified operation.
    /// This is fundamental for gradient synchronization in distributed training.
    pub async fn all_reduce(&mut self, _operation: ReduceOperation) -> Result<()> {
        // Placeholder implementation - in practice this would:
        // 1. Perform collective reduction across all devices
        // 2. Update local shards with reduced values
        // 3. Ensure synchronization across the distributed group

        Ok(())
    }
}

/// Reduction operations for distributed collectives
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOperation {
    /// Sum all values
    Sum,
    /// Average all values
    Mean,
    /// Find maximum value
    Max,
    /// Find minimum value
    Min,
}

impl<T: DataType> AsAny for DistributedStorage<T> {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl<T: DataType> Storage<T> for DistributedStorage<T> {
    fn as_slice(&self) -> &[T] {
        // Distributed storage doesn't have a single contiguous slice
        // This would need to gather data first, which is expensive
        // Return empty slice as a compromise
        &[]
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // Same issue as as_slice
        &mut []
    }

    fn shape(&self) -> &Shape {
        &self.global_shape
    }

    fn strides(&self) -> &[usize] {
        // Distributed tensors don't have simple strides
        &[]
    }

    fn is_contiguous(&self) -> bool {
        // Distributed tensors are never contiguous
        false
    }

    fn as_storage_ref(&self) -> &dyn Storage<T> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dense::DenseStorage, DataType};
    use coeus_dtype::float::F32;

    #[test]
    fn test_distributed_storage_creation() {
        let global_shape = [100, 50];
        let device_ids = vec![DeviceId(0), DeviceId(1), DeviceId(2)];

        let storage = DistributedStorage::<F32>::new(
            &global_shape,
            ShardingStrategy::RowWise,
            &device_ids,
        )
        .unwrap();

        assert_eq!(storage.global_shape().dims(), &[100, 50]);
        assert_eq!(storage.shards().len(), 3); // One shard per device

        // Check that shards are approximately equal size
        let shard_sizes: Vec<usize> = storage.shards().iter().map(|s| s.shape.dims()[0]).collect();
        assert_eq!(shard_sizes, vec![34, 33, 33]); // 100 / 3 = 33*3 + 1
    }

    #[test]
    fn test_column_wise_sharding() {
        let global_shape = [10, 20];
        let device_ids = vec![DeviceId(0), DeviceId(1)];

        let storage = DistributedStorage::<F32>::new(
            &global_shape,
            ShardingStrategy::ColumnWise,
            &device_ids,
        )
        .unwrap();

        assert_eq!(storage.shards().len(), 2);
        assert_eq!(storage.shards()[0].shape.dims(), &[10, 10]); // 20 / 2
        assert_eq!(storage.shards()[1].shape.dims(), &[10, 10]);
    }

    #[test]
    fn test_replicated_sharding() {
        let global_shape = [5, 5];
        let device_ids = vec![DeviceId(0), DeviceId(1)];

        let storage = DistributedStorage::<F32>::new(
            &global_shape,
            ShardingStrategy::Replicated,
            &device_ids,
        )
        .unwrap();

        assert_eq!(storage.shards().len(), 2);
        // All shards should have the full shape
        for shard in storage.shards() {
            assert_eq!(shard.shape.dims(), &[5, 5]);
        }
    }

    #[test]
    fn test_block_wise_sharding() {
        let global_shape = [6, 6];
        let device_ids = vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3)];

        let storage = DistributedStorage::<F32>::new(
            &global_shape,
            ShardingStrategy::BlockWise {
                block_rows: 3,
                block_cols: 3,
            },
            &device_ids,
        )
        .unwrap();

        assert_eq!(storage.shards().len(), 4);
        // All shards should be 3x3 blocks
        for shard in storage.shards() {
            assert_eq!(shard.shape.dims(), &[3, 3]);
        }
    }

    #[test]
    fn test_add_local_shard() {
        let global_shape = [10, 10];
        let device_ids = vec![DeviceId(0)];

        let mut storage = DistributedStorage::<F32>::new(
            &global_shape,
            ShardingStrategy::Replicated,
            &device_ids,
        )
        .unwrap();

        // Create a dense storage for the shard
        let data = vec![F32::new(1.0); 100];
        let dense_storage = DenseStorage::from_vec(data, &[10, 10]).unwrap();

        // Add it as a local shard
        storage
            .add_local_shard(0, StorageVariant::Dense(dense_storage))
            .unwrap();

        assert_eq!(storage.local_shards().len(), 1);
        assert!(storage.get_local_shard(0).is_some());
    }

    #[test]
    fn test_invalid_sharding_configuration() {
        // Test with empty device list
        let result = DistributedStorage::<F32>::new(
            &[10, 10],
            ShardingStrategy::RowWise,
            &[],
        );
        assert!(result.is_err());

        // Test column-wise on 1D tensor
        let result = DistributedStorage::<F32>::new(
            &[10],
            ShardingStrategy::ColumnWise,
            &[DeviceId(0)],
        );
        assert!(result.is_err());
    }
}

// StorageToDense implementation for distributed storage
impl<T: DataType> crate::StorageToDense<T> for DistributedStorage<T> {
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        // For now, create a dense tensor filled with zeros
        // In a full implementation, this would gather all shards from devices
        let size = self.global_shape.size();
        let dense_data = vec![T::zero(); size];
        crate::DenseStorage::from_vec(dense_data, self.global_shape.dims())
    }
}
