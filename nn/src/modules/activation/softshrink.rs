//! Softshrink Activation Function
//!
//! Softshrink(x, λ) = { x - λ, if x > λ
//!                   { x + λ, if x < -λ  
//!                   { 0,     otherwise

use crate::core::error::Result;
use crate::modules::activation::Activation;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{FloatExt, Tensor};

/// Softshrink activation function
///
/// Applies the soft shrinkage function element-wise:
/// - If x > λ: output = x - λ
/// - If x < -λ: output = x + λ
/// - Otherwise: output = 0
#[derive(Clone)]
pub struct Softshrink<T> {
    /// Lambda parameter (threshold)
    pub lambd: T,
    _marker: std::marker::PhantomData<T>,
}

impl<T: DataType + FloatExt + num_traits::FromPrimitive> Softshrink<T> {
    /// Create a new Softshrink with default lambda=0.5
    pub fn new() -> Self {
        Self {
            lambd: T::from_f64(0.5).unwrap_or_else(|| T::zero()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new Softshrink with custom lambda
    pub fn with_lambd(lambd: T) -> Self {
        Self {
            lambd,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: DataType + FloatExt + num_traits::FromPrimitive> Default for Softshrink<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Activation<B, S, T> for Softshrink<T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Copy + PartialOrd,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let data = x.as_slice();
        let lambd = self.lambd;
        let neg_lambd = -lambd;
        
        let result: Vec<T> = data
            .iter()
            .map(|&v| {
                if v > lambd {
                    v - lambd
                } else if v < neg_lambd {
                    v + lambd
                } else {
                    T::zero()
                }
            })
            .collect();

        Tensor::from_vec_with_backend(result, x.shape().dims(), x.backend().clone())
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;

    #[test]
    fn test_softshrink_basic() {
        let softshrink = Softshrink::<Float32>::new();
        let tensor: Tensor<TestBackend, TestStorage, Float32> =
            Tensor::from_vec(vec![Float32(-1.0), Float32(0.0), Float32(1.0)], &[3]).unwrap();
        
        let result = softshrink.forward(&tensor).unwrap();
        let data = result.as_slice();
        
        // x=-1.0: -1.0 < -0.5, so output = -1.0 + 0.5 = -0.5
        assert!((data[0].0 - (-0.5)).abs() < 1e-5);
        // x=0.0: within threshold, output = 0
        assert!((data[1].0 - 0.0).abs() < 1e-5);
        // x=1.0: 1.0 > 0.5, so output = 1.0 - 0.5 = 0.5
        assert!((data[2].0 - 0.5).abs() < 1e-5);
    }
}
