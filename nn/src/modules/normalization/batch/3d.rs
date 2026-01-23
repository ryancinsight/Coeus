//! Batch Normalization for 3D inputs (NCDHW format).

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use crate::modules::normalization::batch::core::BatchNormBase;

/// Batch Normalization layer for 3D inputs (NCDHW format).
#[derive(Debug, Clone)]
pub struct BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Number of features/channels to normalize
    pub num_features: usize,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Running statistics momentum
    pub momentum: f64,
    /// Training mode flag
    pub training: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Scale parameter γ [num_features]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β [num_features]
    pub bias: Parameter<B, S, T>,
    /// Running mean [num_features]
    pub running_mean: std::cell::RefCell<Tensor<B, S, T>>,
    /// Running variance [num_features]
    pub running_var: std::cell::RefCell<Tensor<B, S, T>>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B, S, T> BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new BatchNorm3d layer.
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Result<Self> {
        Self::new_with_backend(B::default(), num_features, eps, momentum, true)
    }

    /// Create a new BatchNorm3d layer with a specific backend and parameters.
    pub fn new_with_backend(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
        track_running_stats: bool,
    ) -> Result<Self> {
        let base = BatchNormBase::new(backend, num_features, eps, momentum, track_running_stats)?;
        Ok(Self {
            num_features: base.num_features,
            eps: base.eps,
            momentum: base.momentum,
            training: base.training,
            track_running_stats: base.track_running_stats,
            weight: base.weight,
            bias: base.bias,
            running_mean: base.running_mean,
            running_var: base.running_var,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get current running mean (for inspection/testing).
    pub fn running_mean(&self) -> Tensor<B, S, T> {
        self.running_mean.borrow().clone()
    }

    /// Get current running variance (for inspection/testing).
    pub fn running_var(&self) -> Tensor<B, S, T> {
        self.running_var.borrow().clone()
    }

    fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) -> Result<()> {
        if !self.track_running_stats {
            return Ok(());
        }

        let momentum_t = T::from(self.momentum).ok_or_else(|| NNError::NumericalError {
            message: format!("momentum ({}) not representable", self.momentum),
        })?;
        let one_minus_momentum =
            T::from(1.0 - self.momentum).ok_or_else(|| NNError::NumericalError {
                message: format!("1.0 - momentum ({}) not representable", 1.0 - self.momentum),
            })?;

        // Update running mean
        {
            let running_mean_tensor = self.running_mean.borrow();
            let running_mean_slice = running_mean_tensor.as_slice();
            let mut updated_mean = Vec::with_capacity(self.num_features);
            for i in 0..self.num_features {
                updated_mean
                    .push(momentum_t * running_mean_slice[i] + one_minus_momentum * batch_mean[i]);
            }
            drop(running_mean_tensor);
            *self.running_mean.borrow_mut() = Tensor::from_vec(updated_mean, &[self.num_features])?;
        }

        // Update running variance
        {
            let running_var_tensor = self.running_var.borrow();
            let running_var_slice = running_var_tensor.as_slice();
            let mut updated_var = Vec::with_capacity(self.num_features);
            for i in 0..self.num_features {
                updated_var
                    .push(momentum_t * running_var_slice[i] + one_minus_momentum * batch_var[i]);
            }
            drop(running_var_tensor);
            *self.running_var.borrow_mut() = Tensor::from_vec(updated_var, &[self.num_features])?;
        }

        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let requires_grad = input.requires_grad();
        let input_dense = input.to_dense_generic()?;
        let input_shape = input_dense.shape().dims();

        if input_shape.len() != 5usize {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Expected 5D input [N, C, D, H, W], got {}D",
                    input_shape.len()
                ),
            });
        }
        if input_shape[1] != self.num_features {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Input channels ({}) must match num_features ({})",
                    input_shape[1], self.num_features
                ),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let depth = input_shape[2];
        let height = input_shape[3];
        let width = input_shape[4];

        let input_data = input_dense.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();
        let eps = T::from(self.eps).unwrap();

        let (batch_mean, batch_var) = if self.training {
            let n_elements = (batch_size * depth * height * width) as f64;
            let n_elements_inv = T::from(1.0 / n_elements).unwrap();

            let mut means = Vec::with_capacity(channels);
            let mut vars = Vec::with_capacity(channels);

            for c in 0..channels {
                let mut sum = T::zero();
                for n in 0..batch_size {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let idx =
                                    (((n * channels + c) * depth + d) * height + h) * width + w;
                                sum = sum + input_data[idx];
                            }
                        }
                    }
                }
                let mean = sum * n_elements_inv;
                means.push(mean);

                let mut var_sum = T::zero();
                for n in 0..batch_size {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let idx =
                                    (((n * channels + c) * depth + d) * height + h) * width + w;
                                let diff = input_data[idx] - mean;
                                var_sum = var_sum + diff * diff;
                            }
                        }
                    }
                }
                vars.push(var_sum * n_elements_inv);
            }

            self.update_running_stats(&means, &vars)?;
            (means, vars)
        } else {
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            (
                running_mean.as_slice().to_vec(),
                running_var.as_slice().to_vec(),
            )
        };

        let mut output_data = Vec::with_capacity(input_data.len());
        for n in 0..batch_size {
            for c in 0..channels {
                let mean = batch_mean[c];
                let std = (batch_var[c] + eps).sqrt();
                let gamma = weight_data[c];
                let beta = bias_data[c];
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = (((n * channels + c) * depth + d) * height + h) * width + w;
                            let val = input_data[idx];
                            output_data.push(gamma * ((val - mean) / std) + beta);
                        }
                    }
                }
            }
        }

        let output_dense = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
            output_data,
            &[batch_size, channels, depth, height, width],
            input.backend().clone(),
        )?;

        let result_data = output_dense.as_slice().to_vec();
        let result_shape = output_dense.shape().dims();
        let result_storage = S::from_vec(result_data, result_shape)?;
        Ok(
            Tensor::from_storage(result_storage, input.backend().clone())
                .requires_grad_(requires_grad),
        )
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn name(&self) -> &str {
        "BatchNorm3d"
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
