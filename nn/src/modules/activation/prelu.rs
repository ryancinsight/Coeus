use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

use super::Activation;

/// PReLU (Parametric ReLU) activation function
///
/// PReLU(x) = max(0, x) + a * min(0, x)
/// where a is a learnable parameter.
#[derive(Debug, Clone)]
pub struct PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Learnable parameter a
    pub weight: Parameter<B, S, T>,
}

impl<B, S, T> PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Copy,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
{
    /// Create a new PReLU activation function
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (1 for shared, or number of channels)
    /// * `init_val` - Initial value for a (default 0.25)
    pub fn new(num_parameters: usize, init_val: Option<T>) -> Self {
        let val = init_val.unwrap_or_else(|| T::from(0.25).unwrap());
        let weight_tensor =
            Tensor::<B, S, T>::from_vec(vec![val; num_parameters], &[num_parameters])
                .unwrap()
                .requires_grad_(true); // Parameters require gradients

        Self {
            weight: Parameter::new(weight_tensor, "weight".to_string()),
        }
    }

    /// Apply PReLU activation
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // PReLU(x) = max(0, x) + a * min(0, x)

        let zero = Tensor::<B, S, T>::zeros_like(x)?;
        let pos = maximum(x, &zero)?;
        let neg = minimum(x, &zero)?;

        // Broadcast weight if necessary
        let weight_tensor = self.weight.data();
        let num_params = weight_tensor.shape().size();

        if num_params == 1 {
            let w_scalar = weight_tensor.to_dense_generic()?.as_slice()[0];
            let neg_scaled = neg.mul_scalar(w_scalar)?;
            Ok(add(&pos, &neg_scaled)?)
        } else {
            // Determine broadcasting shape
            let input_dims = x.shape().dims();
            let rank = input_dims.len();

            let target_shape = if rank >= 2 && input_dims[1] == num_params {
                // Batched input: [N, C, ...] -> reshape weight to [1, C, 1, ...]
                let mut shape = vec![1; rank];
                shape[1] = num_params;
                Some(shape)
            } else if rank >= 1 && input_dims[0] == num_params {
                // Unbatched input: [C, ...] -> reshape weight to [C, 1, ...]
                let mut shape = vec![1; rank];
                shape[0] = num_params;
                Some(shape)
            } else {
                None
            };

            if let Some(shape) = target_shape {
                let shape_isize: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
                let w_reshaped_dense = weight_tensor.reshape(&shape_isize)?;
                let w_reshaped =
                    Tensor::<B, S, T>::from_vec(w_reshaped_dense.as_slice().to_vec(), &shape)?;
                let neg_scaled = mul(&neg, &w_reshaped)?;
                Ok(add(&pos, &neg_scaled)?)
            } else {
                Err(NNError::InvalidInput {
                    message: format!(
                        "PReLU parameter size {} does not match input shape {:?}",
                        num_params, input_dims
                    ),
                })
            }
        }
    }
}

impl<B, S, T> Activation<B, S, T> for PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Copy,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(x)
    }
}

impl<B, S, T> Module<B, S, T> for PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Copy,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone()]
    }

    fn name(&self) -> &str {
        "PReLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
