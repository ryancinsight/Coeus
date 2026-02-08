use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::{ops::arithmetic::*, FloatExt, Tensor};

use super::Activation;

/// Swish-Gated Linear Unit (SwiGLU) activation function
///
/// SwiGLU(x, y) = x * σ(y) where σ is the sigmoid function
/// This is used in modern transformer architectures like PaLM and LLaMA
#[derive(Debug, Clone)]
pub struct SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new SwiGLU activation function
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply SwiGLU activation: x * sigmoid(y)
    ///
    /// # Arguments
    /// * `x` - First input tensor (typically the main activation)
    /// * `y` - Second input tensor (typically the gating signal)
    ///
    /// # Returns
    /// Result containing the SwiGLU activated tensor
    pub fn forward(&self, x: &Tensor<B, S, T>, y: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        if x.shape() != y.shape() {
            return Err(NNError::InvalidInput {
                message: "SwiGLU inputs must have the same shape".to_string(),
            });
        }

        // Compute sigmoid of y: σ(y) = 1 / (1 + exp(-y))
        let sigmoid_y = self.sigmoid(y)?;

        // Compute element-wise multiplication: x * σ(y)
        let result = mul(x, &sigmoid_y)?;

        Ok(result)
    }

    /// Compute sigmoid function: σ(x) = 1 / (1 + exp(-x))
    fn sigmoid(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // sigmoid(x) = 1 / (1 + exp(-x))

        // First compute -x
        let neg_x = neg(x)?;
        // Then compute exp(-x) - need to use elementwise exp
        let exp_neg_x = neg_x.exp();

        // Then compute 1 + exp(-x)
        let one = Tensor::<B, S, T>::ones(x.shape().dims())?;
        let denominator = add(&one, &exp_neg_x)?;

        // Finally compute 1 / (1 + exp(-x))
        let result = div(&one, &denominator)?;

        Ok(result)
    }

    /// Apply SwiGLU activation with automatic gating
    ///
    /// This is a convenience method that splits the input tensor
    /// into two halves and applies SwiGLU with x = first_half, y = second_half
    ///
    /// # Arguments
    /// * `input` - Input tensor with even last dimension size
    ///
    /// # Returns
    /// Result containing the SwiGLU activated tensor
    pub fn forward_split(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let dims = input.shape().dims();
        if dims.is_empty() {
            return Err(NNError::InvalidInput {
                message: "Empty input tensor for SwiGLU forward_split".to_string(),
            });
        }

        let last_dim = *dims.last().unwrap();
        if last_dim == 0 {
            return Err(NNError::InvalidInput {
                message: "Last dimension of input tensor must be > 0 for SwiGLU forward_split"
                    .to_string(),
            });
        }
        if last_dim % 2 != 0 {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Last dimension of input tensor must be even for SwiGLU forward_split (got {})",
                    last_dim
                ),
            });
        }

        let half = last_dim / 2;
        let half_i32 = i32::try_from(half).map_err(|_| NNError::InvalidInput {
            message: format!(
                "SwiGLU forward_split does not support last_dim/2={} exceeding i32::MAX",
                half
            ),
        })?;
        let last_i32 = i32::try_from(last_dim).map_err(|_| NNError::InvalidInput {
            message: format!(
                "SwiGLU forward_split does not support last_dim={} exceeding i32::MAX",
                last_dim
            ),
        })?;

        let dense = input.to_dense_generic()?;
        let rank = dims.len();

        let mut x_slices = Vec::with_capacity(rank);
        let mut y_slices = Vec::with_capacity(rank);
        for _ in 0..rank.saturating_sub(1) {
            x_slices.push((None, None, 1));
            y_slices.push((None, None, 1));
        }
        x_slices.push((Some(0), Some(half_i32), 1));
        y_slices.push((Some(half_i32), Some(last_i32), 1));

        let x_dense = dense.advanced_slice(&x_slices)?;
        let y_dense = dense.advanced_slice(&y_slices)?;

        let backend = dense.backend().clone();
        let x = Tensor::<B, S, T>::from_vec_with_backend(
            x_dense.as_slice().to_vec(),
            x_dense.shape().dims(),
            backend.clone(),
        )?;
        let y = Tensor::<B, S, T>::from_vec_with_backend(
            y_dense.as_slice().to_vec(),
            y_dense.shape().dims(),
            backend,
        )?;

        self.forward(&x, &y)
    }
}

impl<B, S, T> Default for SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Activation<B, S, T> for SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // For trait compatibility, assume split input for SwiGLU
        self.forward_split(x)
    }
}

impl<B, S, T> Module<B, S, T> for SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Module trait forward pass
        self.forward_split(x)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "SwiGLU"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
