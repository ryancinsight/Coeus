//! Advanced Activation Functions for Neural Networks
//!
//! Implementation of state-of-the-art activation functions including:
//! - SwiGLU (Swish-Gated Linear Unit)
//! - GeLU variants
//! - SiLU/Swish
//! - ReLU/PReLU

pub mod gelu;
pub mod prelu;
pub mod relu;
pub mod silu;
pub mod swiglu;

pub use gelu::GeLU;
pub use prelu::PReLU;
pub use relu::ReLU;
pub use silu::SiLU;
pub use swiglu::SwiGLU;

use crate::core::error::Result;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

/// Activation function registry for dynamic activation selection
pub enum ActivationType<T> {
    SwiGLU,
    GeLU,
    SiLU,
    ReLU,
    PReLU(usize, Option<T>),
}

pub struct ActivationFactory<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> ActivationFactory<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Copy,
{
    /// Create an activation function by type
    pub fn create(activation_type: ActivationType<T>) -> Box<dyn Activation<B, S, T>> {
        match activation_type {
            ActivationType::SwiGLU => Box::new(SwiGLU::new()),
            ActivationType::GeLU => Box::new(GeLU::new()),
            ActivationType::SiLU => Box::new(SiLU::new()),
            ActivationType::ReLU => Box::new(ReLU::new()),
            ActivationType::PReLU(num_params, init) => Box::new(PReLU::new(num_params, init)),
        }
    }
}

/// Common activation trait for polymorphism
pub trait Activation<B, S, T>: Send + Sync
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestDataType = Float32;

    #[test]
    fn test_swiglu_basic() {
        let swiglu = SwiGLU::<TestBackend, TestStorage, TestDataType>::new();

        // Create test tensors
        let x_data = vec![
            Float32::new(1.0),
            Float32::new(-1.0),
            Float32::new(2.0),
            Float32::new(-2.0),
        ];
        let y_data = vec![
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(1.0),
        ];

        let x = Tensor::from_vec(x_data, &[2, 2]).unwrap();
        let y = Tensor::from_vec(y_data, &[2, 2]).unwrap();

        let result = swiglu.forward(&x, &y).unwrap();

        // Check that result has correct shape
        assert_eq!(result.shape().dims(), &[2, 2]);

        // SwiGLU(1, 0) = 1 * sigmoid(0) = 1 * 0.5 = 0.5
        // SwiGLU(-1, 0) = -1 * sigmoid(0) = -1 * 0.5 = -0.5
        // SwiGLU(2, 1) = 2 * sigmoid(1) ≈ 2 * 0.731 = 1.462
        // SwiGLU(-2, 1) = -2 * sigmoid(1) ≈ -2 * 0.731 = -1.462

        let result_data = result.as_slice();
        assert!(result_data[0] > Float32::new(0.4) && result_data[0] < Float32::new(0.6)); // ≈ 0.5
        assert!(result_data[1] > Float32::new(-0.6) && result_data[1] < Float32::new(-0.4));
        // ≈ -0.5
    }

    #[test]
    fn test_gelu_approximation() {
        let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();

        // Test with zero input
        let x_data = vec![Float32::new(0.0)];
        let x = Tensor::from_vec(x_data, &[1]).unwrap();

        let result = gelu.forward(&x).unwrap();
        let result_data = result.as_slice();

        // GELU(0) should be approximately 0
        assert!(result_data[0] >= Float32::new(-0.1) && result_data[0] <= Float32::new(0.1));
    }

    #[test]
    fn test_activation_factory() {
        let relu = ActivationFactory::create(ActivationType::ReLU);
        let swiglu = ActivationFactory::create(ActivationType::SwiGLU);

        let x_data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)];
        let x = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(x_data, &[3]).unwrap();

        // Test ReLU
        let relu_result = relu.forward(&x).unwrap();
        let relu_data = relu_result.as_slice();
        assert_eq!(relu_data[0], Float32::new(0.0)); // ReLU(-1) = 0
        assert_eq!(relu_data[1], Float32::new(0.0)); // ReLU(0) = 0
        assert_eq!(relu_data[2], Float32::new(1.0)); // ReLU(1) = 1

        // Test SwiGLU (split mode)
        let swiglu_input: Vec<Float32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -1.0]
            .into_iter()
            .map(Float32::new)
            .collect(); // 2 elements per group
        let swiglu_tensor =
            Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(swiglu_input, &[3, 2])
                .unwrap();
        let _swiglu_result = swiglu.forward(&swiglu_tensor).unwrap();
        // SwiGLU split test would require more complex assertions
    }
}
