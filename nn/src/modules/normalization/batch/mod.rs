//! Batch Normalization layers for neural networks.

use std::cell::RefCell;
use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Batch Normalization layer for 2D inputs (NCHW format).
///
/// Normalizes activations across the batch dimension to stabilize training.
/// During training, uses batch statistics and automatically updates running statistics.
/// During evaluation, uses running statistics for deterministic inference.
///
/// Running statistics are updated automatically during training forward passes using
/// interior mutability (RefCell), requiring no manual intervention.
///
/// Formula:
/// ```text
/// Training mode:
///   batch_mean = Σ(x) / N
///   batch_var = Σ((x - batch_mean)²) / N
///   x_normalized = (x - batch_mean) / √(batch_var + ε)
///   output = γ * x_normalized + β
///
///   # Automatically update running statistics
///   running_mean = momentum * running_mean + (1 - momentum) * batch_mean
///   running_var = momentum * running_var + (1 - momentum) * batch_var
///
/// Evaluation mode:
///   x_normalized = (x - running_mean) / √(running_var + ε)
///   output = γ * x_normalized + β
/// ```
///
/// # Examples
/// ```rust
/// use nn::{BatchNorm2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create BatchNorm2d for 64 channels
/// let mut batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::<Float32>::default(), 64, 1e-5, 0.1).unwrap();
///
/// // Set to training mode
/// <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut batchnorm, true);
///
/// // Input: \[batch_size=2, channels=64, height=32, width=32\]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 64, 32, 32]).unwrap();
///
/// // Output: Same shape, normalized
/// let output = <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&batchnorm, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 32, 32]);
/// ```
#[derive(Debug)]
pub struct BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Number of channels (C)
    pub num_features: usize,
    /// Scale parameter γ \[C\]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β \[C\]
    pub bias: Parameter<B, S, T>,
    /// Running mean \[C\] (interior mutability for automatic updates)
    pub running_mean: RefCell<Tensor<B, S, T>>,
    /// Running variance \[C\] (interior mutability for automatic updates)
    pub running_var: RefCell<Tensor<B, S, T>>,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Running statistics momentum
    pub momentum: f64,
    /// Training mode flag
    pub training: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Phantom data for unused generic parameters
    _phantom: PhantomData<B>,
}

impl<B, S, T> Clone for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: RefCell::new(self.running_mean.borrow().clone()),
            running_var: RefCell::new(self.running_var.borrow().clone()),
            eps: self.eps,
            momentum: self.momentum,
            training: self.training,
            track_running_stats: self.track_running_stats,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> Module<B, S, T> for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        let requires_grad = input.requires_grad();
        let input_dense = input.to_dense_generic()?;

        let input_shape = input_dense.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: "Input must be 4D [N, C, H, W]".to_string(),
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
        let height = input_shape[2];
        let width = input_shape[3];

        let input_data = input_dense.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let eps = T::from(self.eps).ok_or_else(|| NNError::NumericalError {
            message: format!(
                "eps ({}) not representable in dtype {}",
                self.eps,
                T::name()
            ),
        })?;

        let result_dense: Tensor<B, DenseStorage<T>, T> = if self.training {
            let n_elements = batch_size * height * width;
            if n_elements == 0 {
                return Err(NNError::InvalidInput {
                    message: "Input must have non-zero spatial dimensions".to_string(),
                });
            }
            let inv_n_elements = 1.0 / (n_elements as f64);

            let mut batch_mean_f64 = vec![0.0f64; channels];
            let mut batch_var_f64 = vec![0.0f64; channels];

            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mut sum = 0.0f64;
                for n in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let x = input_data[idx].to_f64().ok_or_else(|| {
                                NNError::NumericalError {
                                    message: format!(
                                        "failed converting input value at index {} to f64",
                                        idx
                                    ),
                                }
                            })?;
                            sum += x;
                        }
                    }
                }
                batch_mean_f64[c] = sum * inv_n_elements;
            }

            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mean = batch_mean_f64[c];
                let mut var_sum = 0.0f64;
                for n in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let x = input_data[idx].to_f64().ok_or_else(|| {
                                NNError::NumericalError {
                                    message: format!(
                                        "failed converting input value at index {} to f64",
                                        idx
                                    ),
                                }
                            })?;
                            let diff = x - mean;
                            var_sum += diff * diff;
                        }
                    }
                }
                batch_var_f64[c] = var_sum * inv_n_elements;
            }

            let mut batch_mean = Vec::with_capacity(channels);
            for &mean in &batch_mean_f64 {
                batch_mean.push(T::from(mean).ok_or_else(|| NNError::NumericalError {
                    message: format!(
                        "batch mean ({}) not representable in dtype {}",
                        mean,
                        T::name()
                    ),
                })?);
            }

            let mut batch_var = Vec::with_capacity(channels);
            for &var in &batch_var_f64 {
                batch_var.push(T::from(var).ok_or_else(|| NNError::NumericalError {
                    message: format!(
                        "batch variance ({}) not representable in dtype {}",
                        var,
                        T::name()
                    ),
                })?);
            }

            self.update_running_stats(&batch_mean, &batch_var)?;

            let mut output_data = Vec::with_capacity(input_data.len());
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = batch_mean[c];
                    let std = (batch_var[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let normalized = (input_data[idx] - mean) / std;
                            output_data.push(gamma * normalized + beta);
                        }
                    }
                }
            }

            Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
                output_data,
                &[batch_size, channels, height, width],
                input.backend().clone(),
            )?
        } else {
            let running_mean_data = self.running_mean.borrow();
            let running_var_data = self.running_var.borrow();
            let running_mean_slice = running_mean_data.as_slice();
            let running_var_slice = running_var_data.as_slice();

            let mut output_data = Vec::with_capacity(input_data.len());
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = running_mean_slice[c];
                    let var = running_var_slice[c];
                    let std = (var + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let normalized = (input_data[idx] - mean) / std;
                            output_data.push(gamma * normalized + beta);
                        }
                    }
                }
            }

            Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
                output_data,
                &[batch_size, channels, height, width],
                input.backend().clone(),
            )?
        };

        let result_data = result_dense.as_slice().to_vec();
        let result_shape = result_dense.shape().dims();
        let result_storage = S::from_vec(result_data, result_shape)?;
        Ok(
            Tensor::from_storage(result_storage, input.backend().clone())
                .requires_grad_(requires_grad),
        )
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "BatchNorm2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// * `eps` - Numerical stability constant (default: 1e-5)
    /// * `momentum` - Running statistics momentum (default: 0.1)
    ///
    /// # Weight Initialization
    /// - `weight` (γ): Initialized to 1
    /// - `bias` (β): Initialized to 0
    /// - `running_mean`: Initialized to 0
    /// - `running_var`: Initialized to 1
    pub fn new_with_backend(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "num_features must be > 0".to_string(),
            });
        }
        if eps <= 0.0 {
            return Err(NNError::InvalidConfiguration {
                message: "eps must be > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(NNError::InvalidConfiguration {
                message: "momentum must be in [0.0, 1.0]".to_string(),
            });
        }

        // Initialize weight (γ) to 1
        let weight_data = vec![T::one(); num_features];
        let weight_storage = S::from_vec(weight_data, &[num_features])?;
        let weight_tensor = Tensor::from_storage(weight_storage, backend.clone());
        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        // Initialize bias (β) to 0
        let bias_data = vec![T::zero(); num_features];
        let bias_storage = S::from_vec(bias_data, &[num_features])?;
        let bias_tensor = Tensor::from_storage(bias_storage, backend.clone());
        let bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        // Initialize running_mean to 0 (wrapped in RefCell)
        let running_mean = RefCell::new({
            let mean_data = vec![T::zero(); num_features];
            let mean_storage = S::from_vec(mean_data, &[num_features])?;
            Tensor::from_storage(mean_storage, backend.clone())
        });

        // Initialize running_var to 1 (wrapped in RefCell)
        let running_var = RefCell::new({
            let var_data = vec![T::one(); num_features];
            let var_storage = S::from_vec(var_data, &[num_features])?;
            Tensor::from_storage(var_storage, backend)
        });

        Ok(Self {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            training: true, // Default to training mode - CRITICAL for layer behavior
            track_running_stats: true,
            _phantom: PhantomData,
        })
    }

    /// Create a new BatchNorm2d layer with default CpuBackend.
    ///
    /// Uses default values: eps=1e-5, momentum=0.1, training=true.
    ///
    /// # Arguments
    /// * `num_features` - Number of channels (C)
    /// * `eps` - Numerical stability constant
    /// * `momentum` - Running statistics momentum
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Result<Self>
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), num_features, eps, momentum)
    }

    /// Update running statistics with exponential moving average.
    ///
    /// Update running statistics with exponential moving average.
    ///
    /// This method is called automatically during training forward passes.
    /// Uses interior mutability (RefCell) to update running statistics without requiring &mut self.
    fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) -> Result<()> {
        if !self.track_running_stats {
            return Ok(());
        }

        let momentum_t = T::from(self.momentum).ok_or_else(|| NNError::NumericalError {
            message: format!(
                "momentum ({}) not representable in dtype {}",
                self.momentum,
                T::name()
            ),
        })?;
        let one_minus_momentum =
            T::from(1.0 - self.momentum).ok_or_else(|| NNError::NumericalError {
                message: format!(
                    "1.0 - momentum ({}) not representable in dtype {}",
                    1.0 - self.momentum,
                    T::name()
                ),
            })?;

        // Update running mean using interior mutability
        let new_running_mean = {
            let running_mean_data = self.running_mean.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_mean_data.as_slice()[i]
                        + one_minus_momentum * batch_mean[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_mean.borrow_mut() = Tensor::from_vec(new_running_mean, &[self.num_features])?;

        // Update running var using interior mutability
        let new_running_var = {
            let running_var_data = self.running_var.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_var_data.as_slice()[i] + one_minus_momentum * batch_var[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_var.borrow_mut() = Tensor::from_vec(new_running_var, &[self.num_features])?;
        Ok(())
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

/// Batch Normalization layer for 1D inputs (NCL format).
///
/// Normalizes activations across the batch dimension for sequential data.
/// During training, uses batch statistics and automatically updates running statistics.
/// During evaluation, uses running statistics for deterministic inference.
///
/// Running statistics are updated automatically during training forward passes using
/// interior mutability (RefCell), requiring no manual intervention.
///
/// Formula:
/// ```text
/// Training mode:
///   batch_mean = Σ(x) / (N * L)
///   batch_var = Σ((x - batch_mean)²) / (N * L)
///   x_normalized = (x - batch_mean) / √(batch_var + ε)
///   output = γ * x_normalized + β
///
///   # Automatically update running statistics
///   running_mean = momentum * running_mean + (1 - momentum) * batch_mean
///   running_var = momentum * running_var + (1 - momentum) * batch_var
///
/// Evaluation mode:
///   x_normalized = (x - running_mean) / √(running_var + ε)
///   output = γ * x_normalized + β
/// ```
///
/// # Examples
/// ```rust
/// use nn::{BatchNorm1d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create BatchNorm1d for 128 features
/// let mut batchnorm = BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::<Float32>::default(), 128, 1e-5, 0.1).unwrap();
///
/// // Set to training mode
/// <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut batchnorm, true);
///
/// // Input: [batch_size=32, features=128, sequence_length=100]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[32, 128, 100]).unwrap();
///
/// // Output: Same shape, normalized
/// let output = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&batchnorm, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[32, 128, 100]);
/// ```
///
/// # References
/// - Ioffe & Szegedy (2015): "Batch Normalization: Accelerating Deep Network Training"
/// - Cooijmans et al. (2016): "Recurrent Batch Normalization" - BatchNorm for RNNs
/// - Laurent et al. (2016): "Batch Normalized Recurrent Neural Networks"
#[derive(Debug)]
pub struct BatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Number of features (C)
    pub num_features: usize,
    /// Scale parameter γ [C]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β [C]
    pub bias: Parameter<B, S, T>,
    /// Running mean [C] (interior mutability for automatic updates)
    pub running_mean: RefCell<Tensor<B, S, T>>,
    /// Running variance [C] (interior mutability for automatic updates)
    pub running_var: RefCell<Tensor<B, S, T>>,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Running statistics momentum
    pub momentum: f64,
    /// Training mode flag
    pub training: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Phantom data for unused generic parameters
    _phantom: PhantomData<B>,
}

impl<B, S, T> Clone for BatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: RefCell::new(self.running_mean.borrow().clone()),
            running_var: RefCell::new(self.running_var.borrow().clone()),
            eps: self.eps,
            momentum: self.momentum,
            training: self.training,
            track_running_stats: self.track_running_stats,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> BatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new BatchNorm1d layer with default CpuBackend.
    ///
    /// Uses default values: eps=1e-5, momentum=0.1, training=true.
    ///
    /// # Arguments
    /// * `num_features` - Number of features (C)
    /// * `eps` - Numerical stability constant
    /// * `momentum` - Running statistics momentum
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Result<Self>
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), num_features, eps, momentum)
    }

    /// Create a new BatchNorm1d layer.
    ///
    /// # Arguments
    /// * `num_features` - Number of features (C)
    /// * `eps` - Numerical stability constant (default: 1e-5)
    /// * `momentum` - Running statistics momentum (default: 0.1)
    ///
    /// # Weight Initialization
    /// - `weight` (γ): Initialized to 1
    /// - `bias` (β): Initialized to 0
    /// - `running_mean`: Initialized to 0
    /// - `running_var`: Initialized to 1
    pub fn new_with_backend(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "num_features must be > 0".to_string(),
            });
        }
        if eps <= 0.0 {
            return Err(NNError::InvalidConfiguration {
                message: "eps must be > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(NNError::InvalidConfiguration {
                message: "momentum must be in [0.0, 1.0]".to_string(),
            });
        }

        // Initialize weight (γ) to 1
        let weight_data = vec![T::one(); num_features];
        let weight_storage = S::from_vec(weight_data, &[num_features])?;
        let weight_tensor = Tensor::from_storage(weight_storage, backend.clone());
        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        // Initialize bias (β) to 0
        let bias_data = vec![T::zero(); num_features];
        let bias_storage = S::from_vec(bias_data, &[num_features])?;
        let bias_tensor = Tensor::from_storage(bias_storage, backend.clone());
        let bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        // Initialize running_mean to 0 (wrapped in RefCell)
        let running_mean = RefCell::new({
            let mean_data = vec![T::zero(); num_features];
            let mean_storage = S::from_vec(mean_data, &[num_features])?;
            Tensor::from_storage(mean_storage, backend.clone())
        });

        // Initialize running_var to 1 (wrapped in RefCell)
        let running_var = RefCell::new({
            let var_data = vec![T::one(); num_features];
            let var_storage = S::from_vec(var_data, &[num_features])?;
            Tensor::from_storage(var_storage, backend)
        });

        Ok(Self {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            training: true, // Default to training mode
            track_running_stats: true,
            _phantom: PhantomData,
        })
    }

    /// Update running statistics with exponential moving average.
    ///
    /// This method is called automatically during training forward passes.
    /// Uses interior mutability (RefCell) to update running statistics without requiring &mut self.
    fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) {
        if !self.track_running_stats {
            return;
        }

        let momentum_t = T::from(self.momentum).unwrap();
        let one_minus_momentum = T::from(1.0 - self.momentum).unwrap();

        // Update running mean using interior mutability
        let new_running_mean = {
            let running_mean_data = self.running_mean.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_mean_data.as_slice()[i]
                        + one_minus_momentum * batch_mean[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_mean.borrow_mut() =
            Tensor::from_vec(new_running_mean, &[self.num_features]).unwrap();

        // Update running var using interior mutability
        let new_running_var = {
            let running_var_data = self.running_var.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_var_data.as_slice()[i] + one_minus_momentum * batch_var[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_var.borrow_mut() =
            Tensor::from_vec(new_running_var, &[self.num_features]).unwrap();
    }

    /// Get current running mean (for inspection/testing).
    pub fn running_mean(&self) -> Tensor<B, S, T> {
        self.running_mean.borrow().clone()
    }

    /// Get current running variance (for inspection/testing).
    pub fn running_var(&self) -> Tensor<B, S, T> {
        self.running_var.borrow().clone()
    }

    /// Internal dense computation method for 1D batch normalization.
    ///
    /// This method handles the core batch normalization logic for 3D tensors [N, C, L].
    /// It operates on dense CPU tensors for computation.
    fn compute_batch_norm_1d(
        &self,
        input: Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Input: [N, C, L]
        let input_shape = input.shape().dims();
        if input_shape.len() != 3usize {
            return Err(NNError::InvalidInput {
                message: "Input must be 3D [N, C, L]".to_string(),
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
        let length = input_shape[2];

        let input_data = input.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let eps = T::from(self.eps).unwrap();

        if self.training {
            // Training mode: Use batch statistics
            let n_elements = (batch_size * length) as f64;
            let n_elements_t = T::from(n_elements).unwrap();

            // Compute batch mean and variance for each channel
            let mut batch_mean = vec![T::zero(); channels];
            let mut batch_var = vec![T::zero(); channels];

            // Compute mean: Σ(x) / (N * L)
            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mut sum = T::zero();
                for n in 0..batch_size {
                    for l in 0..length {
                        let idx = (n * channels + c) * length + l;
                        sum = sum + input_data[idx];
                    }
                }
                batch_mean[c] = sum / n_elements_t;
            }

            // Compute variance: Σ((x - mean)²) / (N * L)
            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mean = batch_mean[c];
                let mut var_sum = T::zero();
                for n in 0..batch_size {
                    for l in 0..length {
                        let idx = (n * channels + c) * length + l;
                        let diff = input_data[idx] - mean;
                        var_sum = var_sum + diff * diff;
                    }
                }
                batch_var[c] = var_sum / n_elements_t;
            }

            // Update running statistics automatically
            self.update_running_stats(&batch_mean, &batch_var);

            // Normalize and apply affine transformation
            let mut output_data = Vec::with_capacity(batch_size * channels * length);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = batch_mean[c];
                    let std = (batch_var[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for l in 0..length {
                        let idx = (n * channels + c) * length + l;
                        let normalized = (input_data[idx] - mean) / std;
                        let output_val = gamma * normalized + beta;
                        output_data.push(output_val);
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, length]).map_err(Into::into)
        } else {
            // Evaluation mode: Use running statistics
            let running_mean_data = self.running_mean.borrow();
            let running_var_data = self.running_var.borrow();
            let running_mean_slice = running_mean_data.as_slice();
            let running_var_slice = running_var_data.as_slice();

            let mut output_data = Vec::with_capacity(batch_size * channels * length);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = running_mean_slice[c];
                    let std = (running_var_slice[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for l in 0..length {
                        let idx = (n * channels + c) * length + l;
                        let normalized = (input_data[idx] - mean) / std;
                        let output_val = gamma * normalized + beta;
                        output_data.push(output_val);
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, length]).map_err(Into::into)
        }
    }

    /// Internal dense computation method for 2D batch normalization.
    ///
    /// This method handles the core batch normalization logic for 4D tensors [N, C, H, W].
    /// It operates on dense tensors for computation.
    pub fn compute_batch_norm(
        &self,
        input: Tensor<B, DenseStorage<T>, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>> {
        // Input: [N, C, H, W]
        let input_shape = input.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: "Input must be 4D [N, C, H, W]".to_string(),
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
        let height = input_shape[2];
        let width = input_shape[3];

        let input_data = input.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let eps = T::from(self.eps).ok_or_else(|| NNError::NumericalError {
            message: format!(
                "eps ({}) not representable in dtype {}",
                self.eps,
                T::name()
            ),
        })?;

        if self.training {
            // Training mode: Use batch statistics
            let n_elements = batch_size * height * width;
            if n_elements == 0 {
                return Err(NNError::InvalidInput {
                    message: "Input must have non-zero spatial dimensions".to_string(),
                });
            }
            let inv_n_elements = 1.0 / (n_elements as f64);

            // Compute batch mean and variance for each channel
            let mut batch_mean_f64 = vec![0.0f64; channels];
            let mut batch_var_f64 = vec![0.0f64; channels];

            // Compute mean: Σ(x) / (N * H * W)
            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mut sum = 0.0f64;
                for n in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let x = input_data[idx].to_f64().ok_or_else(|| {
                                NNError::NumericalError {
                                    message: format!(
                                        "failed converting input value at index {} to f64",
                                        idx
                                    ),
                                }
                            })?;
                            sum += x;
                        }
                    }
                }
                batch_mean_f64[c] = sum * inv_n_elements;
            }

            // Compute variance: Σ((x - mean)²) / (N * H * W)
            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mean = batch_mean_f64[c];
                let mut var_sum = 0.0f64;
                for n in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let x = input_data[idx].to_f64().ok_or_else(|| {
                                NNError::NumericalError {
                                    message: format!(
                                        "failed converting input value at index {} to f64",
                                        idx
                                    ),
                                }
                            })?;
                            let diff = x - mean;
                            var_sum += diff * diff;
                        }
                    }
                }
                batch_var_f64[c] = var_sum * inv_n_elements;
            }

            let mut batch_mean = Vec::with_capacity(channels);
            for &mean in &batch_mean_f64 {
                batch_mean.push(T::from(mean).ok_or_else(|| NNError::NumericalError {
                    message: format!(
                        "batch mean ({}) not representable in dtype {}",
                        mean,
                        T::name()
                    ),
                })?);
            }

            let mut batch_var = Vec::with_capacity(channels);
            for &var in &batch_var_f64 {
                batch_var.push(T::from(var).ok_or_else(|| NNError::NumericalError {
                    message: format!(
                        "batch variance ({}) not representable in dtype {}",
                        var,
                        T::name()
                    ),
                })?);
            }

            // Update running statistics automatically
            self.update_running_stats(&batch_mean, &batch_var);

            // Normalize and apply affine transformation
            let mut output_data = Vec::with_capacity(batch_size * channels * height * width);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = batch_mean[c];
                    let std = (batch_var[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let normalized = (input_data[idx] - mean) / std;
                            let output_val = gamma * normalized + beta;
                            output_data.push(output_val);
                        }
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, height, width])
                .map_err(Into::into)
        } else {
            // Evaluation mode: Use running statistics
            let running_mean_data = self.running_mean.borrow();
            let running_var_data = self.running_var.borrow();
            let running_mean_slice = running_mean_data.as_slice();
            let running_var_slice = running_var_data.as_slice();

            let mut output_data = Vec::with_capacity(batch_size * channels * height * width);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = running_mean_slice[c];
                    let var = running_var_slice[c];
                    let std = (var + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((n * channels + c) * height + h) * width + w;
                            let normalized = (input_data[idx] - mean) / std;
                            let output_val = gamma * normalized + beta;
                            output_data.push(output_val);
                        }
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, height, width])
                .map_err(Into::into)
        }
    }
}

impl<B, S, T> Module<B, S, T> for BatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Convert to dense for batch normalization computation
        let input_dense = input.to_dense_generic()?;
        let result_dense = self.compute_batch_norm_1d(input_dense)?;
        // Convert result back to original storage type
        let result_data = result_dense.as_slice().to_vec();
        let result_shape = result_dense.shape().dims();
        let result_storage = S::from_vec(result_data, result_shape)?;
        Ok(Tensor::from_storage(result_storage, B::default()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn zero_grad(&mut self) {
        // No-op: gradients are managed by Parameter
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn name(&self) -> &str {
        "BatchNorm1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}

/// Batch Normalization layer for 3D inputs (NCDHW format).
///
/// Normalizes activations across the batch dimension for spatiotemporal data.
/// During training, uses batch statistics and automatically updates running statistics.
/// During evaluation, uses running statistics for deterministic inference.
///
/// Running statistics are updated automatically during training forward passes using
/// interior mutability (RefCell), requiring no manual intervention.
///
/// Formula:
/// ```text
/// Training mode:
///   batch_mean = Σ(x) / (N * D * H * W)
///   batch_var = Σ((x - batch_mean)²) / (N * D * H * W)
///   x_normalized = (x - batch_mean) / √(batch_var + ε)
///   output = γ * x_normalized + β
///
///   # Automatically update running statistics
///   running_mean = momentum * running_mean + (1 - momentum) * batch_mean
///   running_var = momentum * running_var + (1 - momentum) * batch_var
///
/// Evaluation mode:
///   x_normalized = (x - running_mean) / √(running_var + ε)
///   output = γ * x_normalized + β
/// ```
///
/// # Examples
/// ```rust
/// use nn::{BatchNorm3d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create BatchNorm3d for 64 channels
/// let mut batchnorm = BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::<Float32>::default(), 64, 1e-5, 0.1).unwrap();
///
/// // Set to training mode
/// <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut batchnorm, true);
///
/// // Input: [batch_size=2, channels=64, depth=16, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 64, 16, 32, 32]).unwrap();
///
/// // Output: Same shape, normalized
/// let output = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&batchnorm, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 16, 32, 32]);
/// ```
///
/// # References
/// - Ioffe & Szegedy (2015): "Batch Normalization: Accelerating Deep Network Training"
/// - Tran et al. (2015): "Learning Spatiotemporal Features with 3D Convolutional Networks" (C3D)
/// - Carreira & Zisserman (2017): "Quo Vadis, Action Recognition?" (I3D)
#[derive(Debug)]
pub struct BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Number of channels (C)
    pub num_features: usize,
    /// Scale parameter γ [C]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β [C]
    pub bias: Parameter<B, S, T>,
    /// Running mean [C] (interior mutability for automatic updates)
    pub running_mean: RefCell<Tensor<B, S, T>>,
    /// Running variance [C] (interior mutability for automatic updates)
    pub running_var: RefCell<Tensor<B, S, T>>,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Running statistics momentum
    pub momentum: f64,
    /// Training mode flag
    pub training: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Phantom data for unused generic parameters
    _phantom: PhantomData<B>,
}

impl<B, S, T> Clone for BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: RefCell::new(self.running_mean.borrow().clone()),
            running_var: RefCell::new(self.running_var.borrow().clone()),
            eps: self.eps,
            momentum: self.momentum,
            training: self.training,
            track_running_stats: self.track_running_stats,
            _phantom: PhantomData,
        }
    }
}

impl<B, S, T> BatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new BatchNorm3d layer with default CpuBackend.
    ///
    /// Uses default values: eps=1e-5, momentum=0.1, training=true.
    ///
    /// # Arguments
    /// * `num_features` - Number of channels (C)
    /// * `eps` - Numerical stability constant
    /// * `momentum` - Running statistics momentum
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Result<Self>
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), num_features, eps, momentum)
    }

    /// Create a new BatchNorm3d layer.
    ///
    /// # Arguments
    /// * `num_features` - Number of channels (C)
    /// * `eps` - Numerical stability constant (default: 1e-5)
    /// * `momentum` - Running statistics momentum (default: 0.1)
    ///
    /// # Weight Initialization
    /// - `weight` (γ): Initialized to 1
    /// - `bias` (β): Initialized to 0
    /// - `running_mean`: Initialized to 0
    /// - `running_var`: Initialized to 1
    pub fn new_with_backend(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "num_features must be > 0".to_string(),
            });
        }
        if eps <= 0.0 {
            return Err(NNError::InvalidConfiguration {
                message: "eps must be > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(NNError::InvalidConfiguration {
                message: "momentum must be in [0.0, 1.0]".to_string(),
            });
        }

        // Initialize weight (γ) to 1
        let weight_data = vec![T::one(); num_features];
        let weight_storage = S::from_vec(weight_data, &[num_features])?;
        let weight_tensor = Tensor::from_storage(weight_storage, backend.clone());
        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        // Initialize bias (β) to 0
        let bias_data = vec![T::zero(); num_features];
        let bias_storage = S::from_vec(bias_data, &[num_features])?;
        let bias_tensor = Tensor::from_storage(bias_storage, backend.clone());
        let bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        // Initialize running_mean to 0 (wrapped in RefCell)
        let running_mean = RefCell::new({
            let mean_data = vec![T::zero(); num_features];
            let mean_storage = S::from_vec(mean_data, &[num_features])?;
            Tensor::from_storage(mean_storage, backend.clone())
        });

        // Initialize running_var to 1 (wrapped in RefCell)
        let running_var = RefCell::new({
            let var_data = vec![T::one(); num_features];
            let var_storage = S::from_vec(var_data, &[num_features])?;
            Tensor::from_storage(var_storage, backend)
        });

        Ok(Self {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            momentum,
            training: true, // Default to training mode
            track_running_stats: true,
            _phantom: PhantomData,
        })
    }

    /// Update running statistics with exponential moving average.
    ///
    /// This method is called automatically during training forward passes.
    /// Uses interior mutability (RefCell) to update running statistics without requiring &mut self.
    fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) {
        if !self.track_running_stats {
            return;
        }

        let momentum_t = T::from(self.momentum).unwrap();
        let one_minus_momentum = T::from(1.0 - self.momentum).unwrap();

        // Update running mean using interior mutability
        let new_running_mean = {
            let running_mean_data = self.running_mean.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_mean_data.as_slice()[i]
                        + one_minus_momentum * batch_mean[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_mean.borrow_mut() =
            Tensor::from_vec(new_running_mean, &[self.num_features]).unwrap();

        // Update running var using interior mutability
        let new_running_var = {
            let running_var_data = self.running_var.borrow();
            (0..self.num_features)
                .map(|i| {
                    momentum_t * running_var_data.as_slice()[i] + one_minus_momentum * batch_var[i]
                })
                .collect::<Vec<T>>()
        };
        *self.running_var.borrow_mut() =
            Tensor::from_vec(new_running_var, &[self.num_features]).unwrap();
    }

    /// Get current running mean (for inspection/testing).
    pub fn running_mean(&self) -> Tensor<B, S, T> {
        self.running_mean.borrow().clone()
    }

    /// Get current running variance (for inspection/testing).
    pub fn running_var(&self) -> Tensor<B, S, T> {
        self.running_var.borrow().clone()
    }

    /// Internal dense computation method for 3D batch normalization.
    ///
    /// This method handles the core batch normalization logic for 5D tensors [N, C, D, H, W].
    /// It operates on dense CPU tensors for computation.
    fn compute_batch_norm_3d(
        &self,
        input: Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Input: [N, C, D, H, W]
        let input_shape = input.shape().dims();
        if input_shape.len() != 5usize {
            return Err(NNError::InvalidInput {
                message: "Input must be 5D [N, C, D, H, W]".to_string(),
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

        let input_data = input.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let eps = T::from(self.eps).unwrap();

        if self.training {
            // Training mode: Use batch statistics
            let n_elements = (batch_size * depth * height * width) as f64;
            let n_elements_t = T::from(n_elements).unwrap();

            // Compute batch mean and variance for each channel
            let mut batch_mean = vec![T::zero(); channels];
            let mut batch_var = vec![T::zero(); channels];

            // Compute mean: Σ(x) / (N * D * H * W)
            #[allow(clippy::needless_range_loop)]
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
                batch_mean[c] = sum / n_elements_t;
            }

            // Compute variance: Σ((x - mean)²) / (N * D * H * W)
            #[allow(clippy::needless_range_loop)]
            for c in 0..channels {
                let mean = batch_mean[c];
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
                batch_var[c] = var_sum / n_elements_t;
            }

            // Update running statistics automatically
            self.update_running_stats(&batch_mean, &batch_var);

            // Normalize and apply affine transformation
            let mut output_data =
                Vec::with_capacity(batch_size * channels * depth * height * width);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = batch_mean[c];
                    let std = (batch_var[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let idx =
                                    (((n * channels + c) * depth + d) * height + h) * width + w;
                                let normalized = (input_data[idx] - mean) / std;
                                let output_val = gamma * normalized + beta;
                                output_data.push(output_val);
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, depth, height, width])
                .map_err(Into::into)
        } else {
            // Evaluation mode: Use running statistics
            let running_mean_data = self.running_mean.borrow();
            let running_var_data = self.running_var.borrow();
            let running_mean_slice = running_mean_data.as_slice();
            let running_var_slice = running_var_data.as_slice();

            let mut output_data =
                Vec::with_capacity(batch_size * channels * depth * height * width);
            for n in 0..batch_size {
                for c in 0..channels {
                    let mean = running_mean_slice[c];
                    let std = (running_var_slice[c] + eps).sqrt();
                    let gamma = weight_data[c];
                    let beta = bias_data[c];

                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let idx =
                                    (((n * channels + c) * depth + d) * height + h) * width + w;
                                let normalized = (input_data[idx] - mean) / std;
                                let output_val = gamma * normalized + beta;
                                output_data.push(output_val);
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(output_data, &[batch_size, channels, depth, height, width])
                .map_err(Into::into)
        }
    }
}

impl<S, T> Module<CpuBackend<T>, S, T> for BatchNorm3d<CpuBackend<T>, S, T>
where
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<CpuBackend<T>, S, T>) -> Result<Tensor<CpuBackend<T>, S, T>> {
        // Convert to dense for batch normalization computation
        let input_dense = input.to_dense_generic()?;
        let result_dense = self.compute_batch_norm_3d(input_dense)?;
        // Convert result back to original storage type
        let result_data = result_dense.as_slice().to_vec();
        let result_shape = result_dense.shape().dims();
        let result_storage = S::from_vec(result_data, result_shape)?;
        Ok(Tensor::from_storage(
            result_storage,
            CpuBackend::<T>::default(),
        ))
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, S, T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn modules(&self) -> Vec<&dyn Module<CpuBackend<T>, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        // No-op: gradients are managed by Parameter
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn name(&self) -> &str {
        "BatchNorm3d"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, S, T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;
    use num_traits::ToPrimitive;
    use storage::DenseStorage;

    #[test]
    fn test_batchnorm2d_constructor() {
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();
        assert_eq!(batchnorm.num_features, 64);
        assert_eq!(batchnorm.eps, 1e-5);
        assert_eq!(batchnorm.momentum, 0.1);
        assert!(batchnorm.training);
        assert!(batchnorm.track_running_stats);
    }

    #[test]
    fn test_batchnorm2d_parameter_initialization() {
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                3,
                1e-5,
                0.1,
            )
            .unwrap();

        // Weight (γ) should be initialized to 1
        let weight_data = batchnorm.weight.data().as_slice();
        assert_eq!(weight_data.len(), 3);
        for &w in weight_data {
            assert_eq!(w.to_f64().unwrap(), 1.0);
        }

        // Bias (β) should be initialized to 0
        let bias_data = batchnorm.bias.data().as_slice();
        assert_eq!(bias_data.len(), 3);
        for &b in bias_data {
            assert_eq!(b.to_f64().unwrap(), 0.0);
        }

        // Running mean should be initialized to 0
        let running_mean = batchnorm.running_mean();
        let running_mean_data = running_mean.as_slice();
        assert_eq!(running_mean_data.len(), 3);
        for &m in running_mean_data {
            assert_eq!(m.to_f64().unwrap(), 0.0);
        }

        // Running var should be initialized to 1
        let running_var = batchnorm.running_var();
        let running_var_data = running_var.as_slice();
        assert_eq!(running_var_data.len(), 3);
        for &v in running_var_data {
            assert_eq!(v.to_f64().unwrap(), 1.0);
        }
    }

    #[test]
    fn test_batchnorm2d_forward_training() {
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();

        // Input: [batch_size=2, channels=2, height=2, width=2]
        let input_data = vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1
            5.0, 6.0, 7.0, 8.0, // Batch 1, Channel 0
            9.0, 10.0, 11.0, 12.0, // Batch 1, Channel 1
            13.0, 14.0, 15.0, 16.0,
        ];
        let input_data_f32: Vec<Float32> =
            input_data.iter().map(|&x| Float32::new(x as f32)).collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data_f32,
            &[2, 2, 2, 2],
        )
        .unwrap();

        let output = <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[2, 2, 2, 2]);

        // After normalization, mean should be approximately 0 for each channel
        let output_data = output.as_slice();

        // Channel 0: indices 0-7 (batch 0: 0-3, batch 1: 8-11)
        let channel_0_sum: f64 = [0, 1, 2, 3, 8, 9, 10, 11]
            .iter()
            .map(|&i| output_data[i].to_f64().unwrap())
            .sum();
        let channel_0_mean = channel_0_sum / 8.0;
        assert!(
            (channel_0_mean).abs() < 1e-6,
            "Channel 0 mean should be ~0, got {}",
            channel_0_mean
        );

        // Channel 1: indices 4-7, 12-15 (batch 0: 4-7, batch 1: 12-15)
        let channel_1_sum: f64 = [4, 5, 6, 7, 12, 13, 14, 15]
            .iter()
            .map(|&i| output_data[i].to_f64().unwrap())
            .sum();
        let channel_1_mean = channel_1_sum / 8.0;
        assert!(
            (channel_1_mean).abs() < 1e-6,
            "Channel 1 mean should be ~0, got {}",
            channel_1_mean
        );
    }

    #[test]
    fn test_batchnorm2d_running_stats_update() {
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();

        // Input: [batch_size=2, channels=2, height=2, width=2]
        let input_data = vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1
            5.0, 6.0, 7.0, 8.0, // Batch 1, Channel 0
            9.0, 10.0, 11.0, 12.0, // Batch 1, Channel 1
            13.0, 14.0, 15.0, 16.0,
        ];
        let input_data_f32: Vec<Float32> =
            input_data.iter().map(|&x| Float32::new(x as f32)).collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data_f32,
            &[2, 2, 2, 2],
        )
        .unwrap();

        // Before forward pass, running stats should be initialized
        let running_mean_before = batchnorm.running_mean();
        let running_var_before = batchnorm.running_var();
        assert_eq!(running_mean_before.as_slice()[0].to_f64().unwrap(), 0.0);
        assert_eq!(running_var_before.as_slice()[0].to_f64().unwrap(), 1.0);

        // Forward pass (training mode) - running stats should be updated automatically
        let _output =
            <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
                CpuBackend<Float32>,
                DenseStorage<Float32>,
                Float32,
            >>::forward(&batchnorm, &input)
            .unwrap();

        // After forward pass, running stats should be updated
        let running_mean_after = batchnorm.running_mean();
        let running_var_after = batchnorm.running_var();

        // Running mean should have changed from 0
        assert_ne!(running_mean_after.as_slice()[0].to_f64().unwrap(), 0.0);
        // Running var should have changed from 1
        assert_ne!(running_var_after.as_slice()[0].to_f64().unwrap(), 1.0);
    }

    #[test]
    fn test_batchnorm2d_parameters() {
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();
        let params = <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::parameters(&batchnorm);
        assert_eq!(params.len(), 2); // weight and bias
    }

    #[test]
    fn test_batchnorm2d_invalid_num_features() {
        let result =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                0,
                1e-5,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm2d_invalid_eps() {
        let result =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                0.0,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm2d_invalid_momentum() {
        let result =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                1.5,
            );
        assert!(result.is_err());
    }

    // BatchNorm1d Tests

    #[test]
    fn test_batchnorm1d_constructor() {
        let batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                128,
                1e-5,
                0.1,
            )
            .unwrap();
        assert_eq!(batchnorm.num_features, 128);
        assert_eq!(batchnorm.eps, 1e-5);
        assert_eq!(batchnorm.momentum, 0.1);
        assert!(batchnorm.training);
        assert!(batchnorm.track_running_stats);
    }

    #[test]
    fn test_batchnorm1d_forward_training() {
        let mut batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // Input: [batch_size=2, features=2, length=3]
        // Simple sequential data
        let input_data = vec![
            // Batch 0, Feature 0
            1.0, 2.0, 3.0, // Batch 0, Feature 1
            4.0, 5.0, 6.0, // Batch 1, Feature 0
            7.0, 8.0, 9.0, // Batch 1, Feature 1
            10.0, 11.0, 12.0,
        ];
        let input_data_f32: Vec<Float32> =
            input_data.iter().map(|&x| Float32::new(x as f32)).collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data_f32,
            &[2, 2, 3],
        )
        .unwrap();

        let output = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[2, 2, 3]);

        // In training mode, output should be normalized (mean ≈ 0, std ≈ 1)
        let output_data = output.as_slice();
        assert!(output_data.iter().all(|&x| x.get().is_finite()));
    }

    #[test]
    fn test_batchnorm1d_forward_eval() {
        let mut batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();

        // First, run in training mode to update running stats
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);
        let input_train =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 2, 3])
                .unwrap();
        let _ = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input_train)
        .unwrap();

        // Now switch to eval mode
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, false);

        // Input: [batch_size=1, features=2, length=3]
        let input_eval =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 2, 3])
                .unwrap();
        let output = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input_eval)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[1, 2, 3]);

        // In eval mode, uses running statistics
        let output_data = output.as_slice();
        assert!(output_data.iter().all(|&x| x.get().is_finite()));
    }

    #[test]
    fn test_batchnorm1d_running_stats_update() {
        let mut batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // Input: [batch_size=2, features=2, length=4]
        let input_data = vec![
            // Batch 0, Feature 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Feature 1
            5.0, 6.0, 7.0, 8.0, // Batch 1, Feature 0
            9.0, 10.0, 11.0, 12.0, // Batch 1, Feature 1
            13.0, 14.0, 15.0, 16.0,
        ];
        let input_data_f32: Vec<Float32> =
            input_data.iter().map(|&x| Float32::new(x as f32)).collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data_f32,
            &[2, 2, 4],
        )
        .unwrap();

        // Before forward pass, running stats should be initialized
        let running_mean_before = batchnorm.running_mean();
        assert_eq!(running_mean_before.as_slice()[0].get(), 0.0);
        assert_eq!(running_mean_before.as_slice()[1].get(), 0.0);

        let running_var_before = batchnorm.running_var();
        assert_eq!(running_var_before.as_slice()[0].get(), 1.0);
        assert_eq!(running_var_before.as_slice()[1].get(), 1.0);

        // Forward pass updates running stats
        let _ = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // After forward pass, running stats should be updated
        let running_mean_after = batchnorm.running_mean();
        let running_var_after = batchnorm.running_var();

        // Running mean should be non-zero after update
        assert_ne!(running_mean_after.as_slice()[0].get(), 0.0);
        assert_ne!(running_mean_after.as_slice()[1].get(), 0.0);

        // Running var should be different from initial value
        assert_ne!(running_var_after.as_slice()[0].get(), 1.0);
        assert_ne!(running_var_after.as_slice()[1].get(), 1.0);
    }

    #[test]
    fn test_batchnorm1d_parameters() {
        let batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                128,
                1e-5,
                0.1,
            )
            .unwrap();
        let params = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::parameters(&batchnorm);

        // Should have 2 parameters: weight and bias
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name(), "weight");
        assert_eq!(params[1].name(), "bias");
    }

    #[test]
    fn test_batchnorm1d_train_mode_toggle() {
        let mut batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();

        // Initially in training mode
        assert!(batchnorm.training);

        // Switch to eval mode
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, false);
        assert!(!batchnorm.training);

        // Switch back to training mode
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);
        assert!(batchnorm.training);
    }

    #[test]
    fn test_batchnorm1d_rnn_sequence() {
        let mut batchnorm =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                256,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // RNN hidden states: [batch_size=32, hidden_size=256, sequence_length=50]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[32, 256, 50])
                .unwrap();
        let output = <BatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[32, 256, 50]);
    }

    #[test]
    fn test_batchnorm1d_invalid_num_features() {
        let result =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                0,
                1e-5,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm1d_invalid_eps() {
        let result =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                128,
                0.0,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm1d_invalid_momentum() {
        let result =
            BatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                128,
                1e-5,
                1.5,
            );
        assert!(result.is_err());
    }

    // BatchNorm3d Tests

    #[test]
    fn test_batchnorm3d_constructor() {
        let batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();
        assert_eq!(batchnorm.num_features, 64);
        assert_eq!(batchnorm.eps, 1e-5);
        assert_eq!(batchnorm.momentum, 0.1);
        assert!(batchnorm.training);
        assert!(batchnorm.track_running_stats);
    }

    #[test]
    fn test_batchnorm3d_forward_training() {
        let mut batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // Input: [batch_size=2, channels=2, depth=2, height=2, width=2]
        let input_data = vec![
            // Batch 0, Channel 0, Depth 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 0, Depth 1
            5.0, 6.0, 7.0, 8.0, // Batch 0, Channel 1, Depth 0
            9.0, 10.0, 11.0, 12.0, // Batch 0, Channel 1, Depth 1
            13.0, 14.0, 15.0, 16.0, // Batch 1, Channel 0, Depth 0
            17.0, 18.0, 19.0, 20.0, // Batch 1, Channel 0, Depth 1
            21.0, 22.0, 23.0, 24.0, // Batch 1, Channel 1, Depth 0
            25.0, 26.0, 27.0, 28.0, // Batch 1, Channel 1, Depth 1
            29.0, 30.0, 31.0, 32.0,
        ];
        let input_data_f32: Vec<Float32> =
            input_data.iter().map(|&x| Float32::new(x as f32)).collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data_f32,
            &[2, 2, 2, 2, 2],
        )
        .unwrap();

        let output = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[2, 2, 2, 2, 2]);

        // In training mode, output should be normalized
        let output_data = output.as_slice();
        assert!(output_data.iter().all(|&x| x.get().is_finite()));
    }

    #[test]
    fn test_batchnorm3d_forward_eval() {
        let mut batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();

        // First, run in training mode to update running stats
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);
        let input_train =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 2, 2, 2, 2])
                .unwrap();
        let _ = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input_train)
        .unwrap();

        // Now switch to eval mode
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, false);

        // Input: [batch_size=1, channels=2, depth=2, height=2, width=2]
        let input_eval =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 2, 2, 2, 2])
                .unwrap();
        let output = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input_eval)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[1, 2, 2, 2, 2]);

        // In eval mode, uses running statistics
        let output_data = output.as_slice();
        assert!(output_data.iter().all(|&x| x.get().is_finite()));
    }

    #[test]
    fn test_batchnorm3d_running_stats_update() {
        let mut batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                2,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // Input: [batch_size=2, channels=2, depth=2, height=2, width=2]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 2, 2, 2, 2])
                .unwrap();

        // Before forward pass, running stats should be initialized
        let running_mean_before = batchnorm.running_mean();
        assert_eq!(running_mean_before.as_slice()[0].get(), 0.0);
        assert_eq!(running_mean_before.as_slice()[1].get(), 0.0);

        let running_var_before = batchnorm.running_var();
        assert_eq!(running_var_before.as_slice()[0].get(), 1.0);
        assert_eq!(running_var_before.as_slice()[1].get(), 1.0);

        // Forward pass updates running stats
        let _ = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // After forward pass, running stats should be updated
        let running_mean_after = batchnorm.running_mean();
        let running_var_after = batchnorm.running_var();

        // Running mean should be non-zero after update (momentum * 0 + (1-momentum) * batch_mean)
        assert_ne!(running_mean_after.as_slice()[0].get(), 0.0);
        assert_ne!(running_mean_after.as_slice()[1].get(), 0.0);

        // Running var should be different from initial value
        assert_ne!(running_var_after.as_slice()[0].get(), 1.0);
        assert_ne!(running_var_after.as_slice()[1].get(), 1.0);
    }

    #[test]
    fn test_batchnorm3d_parameters() {
        let batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();
        let params = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::parameters(&batchnorm);

        // Should have 2 parameters: weight and bias
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name(), "weight");
        assert_eq!(params[1].name(), "bias");
    }

    #[test]
    fn test_batchnorm3d_train_mode_toggle() {
        let mut batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();

        // Initially in training mode
        assert!(batchnorm.training);

        // Switch to eval mode
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, false);
        assert!(!batchnorm.training);

        // Switch back to training mode
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);
        assert!(batchnorm.training);
    }

    #[test]
    fn test_batchnorm3d_video_classification() {
        let mut batchnorm =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                0.1,
            )
            .unwrap();
        <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::train(&mut batchnorm, true);

        // Video data: [batch_size=8, channels=64, depth=16, height=32, width=32]
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[
            8, 64, 16, 32, 32,
        ])
        .unwrap();
        let output = <BatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >>::forward(&batchnorm, &input)
        .unwrap();

        // Output shape should match input shape
        assert_eq!(output.shape().dims(), &[8, 64, 16, 32, 32]);
    }

    #[test]
    fn test_batchnorm3d_invalid_num_features() {
        let result =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                0,
                1e-5,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm3d_invalid_eps() {
        let result =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                0.0,
                0.1,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm3d_invalid_momentum() {
        let result =
            BatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::<Float32>::default(),
                64,
                1e-5,
                1.5,
            );
        assert!(result.is_err());
    }
}
