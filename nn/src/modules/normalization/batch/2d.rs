//! Batch Normalization for 2D inputs (NCHW format).

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use crate::modules::normalization::batch::core::BatchNormBase;

/// Batch Normalization layer for 2D inputs (NCHW format).
#[derive(Debug, Clone)]
pub struct BatchNorm2d<B, S, T>
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

impl<B, S, T> BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new BatchNorm2d layer.
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Result<Self> {
        Self::new_with_backend(B::default(), num_features, eps, momentum, true)
    }

    /// Create a new BatchNorm2d layer with a specific backend and parameters.
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
}

impl<B, S, T> Module<B, S, T> for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
            return Err(NNError::ShapeMismatch {
                operation: "BatchNorm2d forward".to_string(),
                expected: vec![0, 0, 0, 0],
                actual: input_shape.to_vec(),
            });
        }

        let channels = input_shape[1];
        if channels != self.num_features {
             return Err(NNError::ShapeMismatch {
                operation: "BatchNorm2d features".to_string(),
                expected: vec![0, self.num_features, 0, 0],
                actual: input_shape.to_vec(),
            });
        }

        let requires_grad = input.requires_grad();

        let output = if self.training {
            let input_dense = input.to_dense_generic()?;
            let input_data = input_dense.as_slice();
            let batch_size = input_shape[0];
            let spatial_size = input_shape[2] * input_shape[3];

            let mut batch_means = Vec::with_capacity(channels);
            let mut batch_vars = Vec::with_capacity(channels);

            for c in 0..channels {
                let mut sum = T::zero();
                let mut sq_sum = T::zero();
                let count = batch_size * spatial_size;

                for n in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = (n * channels + c) * spatial_size + s;
                        let val = input_data[idx];
                        sum = sum + val;
                        sq_sum = sq_sum + val * val;
                    }
                }

                let mean = sum / T::from(count as f64).unwrap();
                let var = sq_sum / T::from(count as f64).unwrap() - mean * mean;

                batch_means.push(mean);
                batch_vars.push(var);
            }

            if self.track_running_stats {
                self.update_running_stats(&batch_means, &batch_vars)?;
            }

            // Using functional::batch_norm which recalculates batch stats for training
            crate::functional::batch_norm(
                input,
                Some(self.weight.data()),
                Some(self.bias.data()),
                self.eps,
            )?
        } else {
            // Evaluation mode uses running statistics
            let input_dense = input.to_dense_generic()?;
            let input_data = input_dense.as_slice();
            let batch_size = input_shape[0];
            let spatial_size = input_shape[2] * input_shape[3];
            
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            let mean_slice = running_mean.as_slice();
            let var_slice = running_var.as_slice();
            
            let weight_slice = self.weight.data().as_slice();
            let bias_slice = self.bias.data().as_slice();
            let eps_t = T::from(self.eps).unwrap();

            let mut output_data = Vec::with_capacity(input_data.len());
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = mean_slice[c];
                    let var = var_slice[c];
                    let std = (var + eps_t).sqrt();
                    let w = weight_slice[c];
                    let b = bias_slice[c];

                    for s in 0..spatial_size {
                        let idx = (n * channels + c) * spatial_size + s;
                        let val = input_data[idx];
                        let normalized = (val - mean) / std;
                        output_data.push(normalized * w + b);
                    }
                }
            }
            Tensor::from_vec_with_backend(output_data, input_shape, input.backend().clone())?
        };

        Ok(output.requires_grad_(requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn name(&self) -> &str {
        "BatchNorm2d"
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
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
