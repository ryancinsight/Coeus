//! Backend dispatch system for tensor operations.
//!
//! This module implements efficient backend dispatching using associated types pattern
//! for compile-time resolution of tensor operations across different backends.
//!
//! Uses the Backend trait's Clone bounds established in sprint MS-43.

use crate::{
    Backend, DataType, DenseStorage, Result, Storage, StorageFromVec, StorageToDense, Tensor,
};

/// Backend operations dispatcher using associated types pattern.
///
/// This trait enables compile-time dispatch of operations to specific backends
/// based on the associated types defined in the Backend trait.
pub trait TensorBackendDispatcher<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone,
    T: DataType,
{
    /// Dispatch tensor addition operation
    fn dispatch_add(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>;

    /// Dispatch tensor multiplication operation
    fn dispatch_mul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>;

    /// Dispatch matrix multiplication between tensors
    fn dispatch_matmul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>;

    /// Dispatch ReLU activation
    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        T: PartialOrd + Default,
        S: StorageToDense<T>;

    /// Dispatch sum reduction
    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>;

    /// Dispatch cross-backend tensor transfer
    fn dispatch_to_backend<NewB>(
        &self,
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>,
        T: Clone;
}

/// Default implementation for any backend that implements the required operations
impl<B, S, T> TensorBackendDispatcher<B, S, T> for B
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType
        + Clone
        + Copy
        + num_traits::Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
{
    fn dispatch_add(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
    {
        // Use the Backend trait's associated types for type-safe dispatch
        // Convert to dense for backend operations if needed
        let lhs_dense = lhs.to_dense_generic()?;
        let rhs_dense = rhs.to_dense_generic()?;

        let result_storage = self.add_dense(&lhs_dense.storage, &rhs_dense.storage)?;

        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_mul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
    {
        let lhs_dense = lhs.to_dense_generic()?;
        let rhs_dense = rhs.to_dense_generic()?;

        let result_storage = self.mul_dense(&lhs_dense.storage, &rhs_dense.storage)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_matmul(
        &self,
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
    {
        let lhs_dense = lhs.to_dense_generic()?;
        let rhs_dense = rhs.to_dense_generic()?;

        let result_storage = self.matmul_dense(&lhs_dense.storage, &rhs_dense.storage)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_relu(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        T: PartialOrd + Default,
        S: StorageToDense<T>,
    {
        let input_dense = input.to_dense_generic()?;
        let result_storage = self.relu_dense(&input_dense.storage)?;
        Ok(Tensor::from_storage(result_storage, self.clone()))
    }

    fn dispatch_sum(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        S: StorageToDense<T>,
    {
        let input_dense = input.to_dense_generic()?;
        let sum_value = self.sum_dense(&input_dense.storage)?;

        // Create scalar tensor with sum result
        let scalar_data = vec![sum_value];
        let scalar_storage =
            DenseStorage::from_vec(scalar_data, &[1]).map_err(crate::TensorError::StorageError)?;

        Ok(Tensor::from_storage(scalar_storage, self.clone()))
    }

    fn dispatch_to_backend<NewB>(
        &self,
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        S: StorageToDense<T>,
        T: Clone,
    {
        tensor.to_backend(target_backend)
    }
}

/// High-level dispatch interface for tensor operations.
///
/// Provides a clean API for backend-agnostic tensor operations.
/// Uses the associated types pattern for efficient dispatch.
pub struct TensorDispatcher;

impl TensorDispatcher {
    /// Dispatch addition operation between tensors
    pub fn add<B, S, T>(
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        lhs.backend().dispatch_add(lhs, rhs)
    }

    /// Dispatch multiplication operation between tensors
    pub fn mul<B, S, T>(
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        lhs.backend().dispatch_mul(lhs, rhs)
    }

    /// Dispatch matrix multiplication between tensors
    pub fn matmul<B, S, T>(
        lhs: &Tensor<B, S, T>,
        rhs: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        lhs.backend().dispatch_matmul(lhs, rhs)
    }

    /// Dispatch ReLU activation
    pub fn relu<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + PartialOrd
            + Default,
    {
        input.backend().dispatch_relu(input)
    }

    /// Dispatch sum reduction
    pub fn sum<B, S, T>(input: &Tensor<B, S, T>) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType
            + Clone
            + Copy
            + num_traits::Zero
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>,
    {
        input.backend().dispatch_sum(input)
    }

    /// Dispatch cross-backend tensor transfer
    pub fn to_backend<B, S, T, NewB>(
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        B: Backend<Data = T> + Clone + Default + TensorBackendDispatcher<B, S, T>,
        S: Storage<T> + Clone + StorageToDense<T> + 'static,
        T: DataType + Clone,
    {
        tensor.backend().dispatch_to_backend(tensor, target_backend)
    }
}

/// Memory transfer operations for cross-backend tensor sharing.
///
/// Implements distributed tensor sharing via Clone bounds as required for sprint MS-44.
pub struct MemoryTransfer;

impl MemoryTransfer {
    /// Transfer tensor between backends with potential zero-copy operations
    pub fn transfer<B, S, T, NewB>(
        tensor: &Tensor<B, S, T>,
        target_backend: NewB,
    ) -> Result<Tensor<NewB, DenseStorage<T>, T>>
    where
        NewB: Backend<Data = T> + Clone + Send + Sync,
        B: Backend + Clone,
        S: Storage<T> + Clone + StorageToDense<T>,
        T: DataType + Clone,
    {
        // Use Clone bounds for efficient transfer
        // Backends can implement zero-copy transfers via Clone trait
        tensor.to_backend(target_backend)
    }

    /// Check if backends support zero-copy transfer between them
    pub fn can_zero_copy_transfer<B, NewB>(source_backend: &B, target_backend: &NewB) -> bool
    where
        B: Backend,
        NewB: Backend,
    {
        // Check device compatibility (same device type and memory space)
        // This is a placeholder - actual implementation depends on backend capabilities
        source_backend.device_name() == target_backend.device_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, DenseStorage, Tensor};
    use coeus_dtype::float::Float32;

    #[test]
    fn test_dispatcher_add() {
        let lhs_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let rhs_data = vec![Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];

        let lhs: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(lhs_data, &[3]).unwrap();
        let rhs: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(rhs_data, &[3]).unwrap();

        let result = TensorDispatcher::add(&lhs, &rhs).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.as_slice()[0].get(), 5.0);
        assert_eq!(result.as_slice()[1].get(), 7.0);
        assert_eq!(result.as_slice()[2].get(), 9.0);
    }

    #[test]
    fn test_backend_supports_operation() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5]).unwrap();
        assert!(tensor.backend_supports("arithmetic"));
    }

        use coeus_backend::DeviceInfo;
    fn test_device_access() {
        use coeus_backend::DeviceInfo;
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3]).unwrap();
        let device = tensor.device();
        assert_eq!(device.name(), "cpu");
    }
}
