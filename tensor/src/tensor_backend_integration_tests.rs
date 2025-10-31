//! Tensor-backend integration tests for sprint MS-44.
//!
//! Tests tensor operations that require Clone bounds, memory transfer operations,
//! and backend-specific optimizations using associated types.


/// Tests for tensor.clone() using new Backend clone bounds
#[cfg(test)]
mod clone_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_tensor_backend_clone() {
        let original = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        ).unwrap();

        let cloned = original.backend_clone();

        assert_eq!(original.shape().dims(), cloned.shape().dims());
        assert_eq!(original.as_slice(), cloned.as_slice());
        assert_eq!(original.requires_grad(), cloned.requires_grad());
    }

    #[test]
    fn test_tensor_backend_clone_independence() {
        let original = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        let mut cloned = original.backend_clone();

        // Modify cloned tensor
        cloned.as_mut_slice()[0] = Float32::new(99.0);

        // Original should remain unchanged
        assert_eq!(original.as_slice()[0].get(), 1.0);
        assert_eq!(cloned.as_slice()[0].get(), 99.0);
    }
}

/// Tests for tensor.to_backend() using Backend trait
#[cfg(test)]
mod to_backend_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_tensor_to_same_backend() {
        let cpu_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        ).unwrap();

        let target_backend = CpuBackend::new();
        let result = cpu_tensor.to_backend(target_backend).unwrap();

        assert_eq!(cpu_tensor.shape().dims(), result.shape().dims());
        assert_eq!(cpu_tensor.as_slice(), result.as_slice());
    }

    #[test]
    fn test_tensor_to_backend_data_integrity() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let cpu_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            data, &[2, 2],
        ).unwrap();

        let target_backend = CpuBackend::new();
        let transferred = cpu_tensor.to_backend(target_backend).unwrap();

        assert_eq!(cpu_tensor.shape().dims(), transferred.shape().dims());
        assert_eq!(cpu_tensor.as_slice(), transferred.as_slice());
    }

    #[test]
    fn test_tensor_device_info_access() {
        use backend::DeviceInfo;

        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();

        let device = tensor.device();
        assert_eq!(device.name(), "cpu");
    }

    #[test]
    fn test_tensor_backend_support_check() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[3]).unwrap();

        assert!(tensor.backend_supports("arithmetic"));
        // CPU backend should support all basic operations in our implementation
    }
}

/// Tests for TensorDispatcher using associated types pattern
#[cfg(test)]
mod dispatcher_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_tensor_dispatcher_add() {
        let lhs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        let rhs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(4.0)],
            &[2],
        ).unwrap();

        let result = TensorDispatcher::add(&lhs, &rhs).unwrap();

        assert_eq!(result.as_slice()[0].get(), 4.0);
        assert_eq!(result.as_slice()[1].get(), 6.0);
    }

    #[test]
    fn test_tensor_dispatcher_mul() {
        let lhs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        ).unwrap();

        let rhs = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(4.0), Float32::new(5.0)],
            &[2],
        ).unwrap();

        let result = TensorDispatcher::mul(&lhs, &rhs).unwrap();

        assert_eq!(result.as_slice()[0].get(), 8.0);
        assert_eq!(result.as_slice()[1].get(), 15.0);
    }

    #[test]
    fn test_tensor_dispatcher_sum() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        ).unwrap();

        let result = TensorDispatcher::sum(&tensor).unwrap();

        assert_eq!(result.as_slice()[0].get(), 6.0);
        assert_eq!(result.shape().dims(), &[1]);
    }

    #[test]
    fn test_tensor_dispatcher_to_backend() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        let target_backend = CpuBackend::new();
        let result = TensorDispatcher::to_backend(&tensor, target_backend).unwrap();

        assert_eq!(tensor.as_slice(), result.as_slice());
    }
}

/// Tests for MemoryTransfer operations using Clone bounds
#[cfg(test)]
mod memory_transfer_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_memory_transfer_same_backend() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        ).unwrap();

        let target_backend = CpuBackend::new();
        let transferred = MemoryTransfer::transfer(&tensor, target_backend).unwrap();

        assert_eq!(tensor.as_slice(), transferred.as_slice());
        assert_eq!(tensor.shape().dims(), transferred.shape().dims());
    }

    #[test]
    fn test_memory_transfer_can_zero_copy() {
        let source_backend = CpuBackend::<Float32>::new();
        let target_backend = CpuBackend::<Float32>::new();

        // Same backend type should support zero-copy in some cases
        let can_zero_copy = MemoryTransfer::can_zero_copy_transfer(&source_backend, &target_backend);
        assert!(can_zero_copy); // CPU to CPU should be zero-copy capable
    }
}

/// Integration tests for distributed tensor sharing using Clone bounds
#[cfg(test)]
mod distributed_sharing_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_distributed_tensor_clone() {
        // Simulate distributed tensor sharing scenario
        let original = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)],
            &[2, 2],
        ).unwrap();

        // Multiple clones representing distributed sharing
        let clone1 = original.backend_clone();
        let clone2 = original.backend_clone();

        assert_eq!(original.as_slice(), clone1.as_slice());
        assert_eq!(original.as_slice(), clone2.as_slice());

        // All tensors remain independent
        assert_eq!(original.as_slice()[0].get(), 1.0);
        assert_eq!(clone1.as_slice()[0].get(), 1.0);
        assert_eq!(clone2.as_slice()[0].get(), 1.0);
    }

    #[test]
    fn test_tensor_sharing_with_gradients() {
        let original = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        ).unwrap().requires_grad_(true);

        let shared = original.backend_clone();

        // Both tensors should have gradient tracking enabled
        assert!(original.requires_grad());
        assert!(shared.requires_grad());

        // Set gradients on original
        let grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.1), Float32::new(0.2)],
            &[2],
        ).unwrap();
        original.set_grad(grad).unwrap();

        // Shared tensor shares gradient storage with original for autograd compatibility
        assert!(shared.grad().is_ok()); // Gradient is now available on shared tensor
        assert!(original.grad().is_ok()); // Gradient was set on original
    }
}

/// Tests for backend-specific optimizations using associated types
#[cfg(test)]
mod backend_optimization_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_backend_device_capabilities() {
        use backend::DeviceInfo;

        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10]).unwrap();

        let device = tensor.device();
        assert_eq!(device.name(), "cpu");

        // CPU backend should be available
        assert!(device.is_available());

        // Get compute units (cores)
        let compute_units = device.compute_units();
        assert!(compute_units > 0);
    }

    #[test]
    fn test_backend_associated_types() {
        use backend::Backend;

        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3]).unwrap();

        // Test that associated types work correctly
        let backend = tensor.backend();
        assert_eq!(backend.device_name(), "cpu");

        // Test device info access
        let device_info = backend.device_info();
        assert_eq!(device_info.name(), "cpu");
    }

    #[test]
    fn test_tensor_backend_interoperability() {
        // Create tensor with one backend
        let cpu_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        ).unwrap();

        // Verify backend operations work
        assert!(cpu_tensor.backend_supports("arithmetic"));

        // Test backend transfer (even to same backend)
        let same_backend_tensor = cpu_tensor.to_backend(CpuBackend::new()).unwrap();
        assert_eq!(cpu_tensor.as_slice(), same_backend_tensor.as_slice());
    }
}

/// Performance benchmark tests (compile-time checks for optimizations)
#[cfg(test)]
mod performance_tests {
    use super::*;
    use crate::{Tensor, CpuBackend, DenseStorage};
    use crate::tensor_backend_dispatch::{TensorDispatcher, MemoryTransfer};
    use dtype::float::Float32;

    #[test]
    fn test_clone_performance_bounds() {
        // Test that Clone bounds allow efficient tensor cloning
        let data: Vec<Float32> = (0..1000).map(|i| Float32::new(i as f32)).collect();
        let large_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            data, &[1000],
        ).unwrap();

        // Clone should work efficiently due to Backend: Clone bounds
        let cloned = large_tensor.backend_clone();
        assert_eq!(large_tensor.shape().dims(), cloned.shape().dims());
        assert_eq!(large_tensor.len(), cloned.len());
    }

    #[test]
    fn test_dispatcher_static_dispatch() {
        // Test that static dispatch works (compile-time verification)
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[1],
        ).unwrap();

        // These operations should dispatch at compile time
        let _summed = TensorDispatcher::sum(&tensor).unwrap();
        let _added = TensorDispatcher::add(&tensor, &tensor).unwrap();
    }
}
