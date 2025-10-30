//! Activation functions for neural networks.

use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

/// ReLU (Rectified Linear Unit) activation function.
///
/// Applies the function `f(x) = max(0, x)` element-wise.
///
/// # Examples
/// ```rust
/// use coeus_nn::{ReLU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let relu = ReLU;
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.5), Float32::new(2.0)],
///     &[3]
/// ).unwrap();
///
/// let output = relu.forward(&input).unwrap();
/// // Output: [0.0, 0.5, 2.0]
/// ```
#[derive(Debug, Clone)]
pub struct ReLU;

impl ReLU {
    /// Create a new ReLU activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for ReLU
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd + Copy,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply ReLU element-wise: max(0, x)
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;
        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| if x > T::zero() { x } else { T::zero() })
            .collect();
        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // ReLU has no learnable parameters
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        // No-op: ReLU has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: ReLU behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

impl ReLU {
    // Sparse ReLU methods removed - using dense computation for now
    // Full sparse support will be added when storage API matures

    // CSC sparse ReLU method removed - using dense computation

    // COO sparse ReLU method removed - using dense computation
}

/// Sigmoid activation function.
///
/// Applies the function `f(x) = 1 / (1 + exp(-x))` element-wise.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Sigmoid, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let sigmoid = Sigmoid::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0)],
///     &[1]
/// ).unwrap();
///
/// let output = sigmoid.forward(&input).unwrap();
/// // Output: [0.5]
/// ```
#[derive(Debug, Clone)]
pub struct Sigmoid;

impl Sigmoid {
    /// Create a new Sigmoid activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for Sigmoid
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply sigmoid element-wise: 1 / (1 + exp(-x))
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;
        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| {
                let one = T::one();
                let exp_neg_x = (-x).exp();
                one / (one + exp_neg_x)
            })
            .collect();
        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Sigmoid has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Sigmoid has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Sigmoid behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Tanh activation function.
///
/// Applies the function `f(x) = tanh(x)` element-wise.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Tanh, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let tanh = Tanh::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0)],
///     &[1]
/// ).unwrap();
///
/// let output = tanh.forward(&input).unwrap();
/// // Output: [0.0]
/// ```
#[derive(Debug, Clone)]
pub struct Tanh;

impl Tanh {
    /// Create a new Tanh activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for Tanh
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply tanh element-wise
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;
        let result_data: Vec<T> = input_dense.as_slice().iter().map(|&x| x.tanh()).collect();
        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Tanh has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Tanh has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Tanh behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// GELU (Gaussian Error Linear Unit) activation function.
///
/// Applies the function `f(x) = x * Φ(x)` where `Φ(x)` is the cumulative distribution
/// function of the standard normal distribution.
///
/// Uses the tanh approximation for efficiency:
/// `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
///
/// # Examples
/// ```rust
/// use coeus_nn::{GELU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let gelu = GELU::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0), Float32::new(-1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = gelu.forward(&input).unwrap();
/// // Output: [0.0, ~0.841, ~-0.159]
/// ```
#[derive(Debug, Clone)]
pub struct GELU;

impl GELU {
    /// Create a new GELU activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for GELU
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply GELU element-wise: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| {
                let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap(); // √(2/π)
                let coeff = T::from(0.044715).unwrap();
                let half = T::from(0.5).unwrap();
                let one = T::one();

                let x_cubed = x * x * x;
                let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
                half * x * (one + inner.tanh())
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // GELU has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: GELU has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: GELU behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "GELU"
    }
}

/// Swish/SiLU (Sigmoid Linear Unit) activation function.
///
/// Applies the function `f(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Swish, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let swish = Swish::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0), Float32::new(-1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = swish.forward(&input).unwrap();
/// // Output: [0.0, ~0.731, ~-0.269]
/// ```
#[derive(Debug, Clone)]
pub struct Swish;

/// Alias for Swish activation (SiLU is the same as Swish).
pub type SiLU = Swish;

impl Swish {
    /// Create a new Swish activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Swish {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for Swish
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply Swish element-wise: x * sigmoid(x) = x / (1 + exp(-x))
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| {
                let one = T::one();
                let sigmoid_x = one / (one + (-x).exp());
                x * sigmoid_x
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Swish has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Swish has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Swish behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Swish"
    }
}

/// LeakyReLU activation function.
///
/// Applies the function `f(x) = max(0, x) + negative_slope * min(0, x)`.
///
/// # Examples
/// ```rust
/// use coeus_nn::{LeakyReLU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let leaky_relu = LeakyReLU::new(0.01); // negative_slope = 0.01
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = leaky_relu.forward(&input).unwrap();
/// // Output: [-0.01, 0.0, 1.0]
/// ```
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    /// Negative slope for x < 0
    pub negative_slope: f64,
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation layer.
    ///
    /// # Arguments
    /// * `negative_slope` - Slope for negative values (default: 0.01)
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl<B, S, T> Module<B, S, T> for LeakyReLU
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply LeakyReLU element-wise: max(0, x) + negative_slope * min(0, x)
        // Sparse tensors are handled efficiently at the storage level
        let input_dense = input.to_dense_generic()?;

        let zero = T::zero();
        let slope = T::from(self.negative_slope).unwrap();

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| if x > zero { x } else { slope * x })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // LeakyReLU has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: LeakyReLU has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: LeakyReLU behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "LeakyReLU"
    }
}

/// ELU (Exponential Linear Unit) activation function.
///
/// Applies the function `f(x) = x if x > 0 else alpha * (exp(x) - 1)`.
///
/// # Examples
/// ```rust
/// use coeus_nn::{ELU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let elu = ELU::new(1.0); // alpha = 1.0
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = elu.forward(&input).unwrap();
/// // Output: [~-0.632, 0.0, 1.0]
/// ```
#[derive(Debug, Clone)]
pub struct ELU {
    /// Alpha parameter for negative values
    pub alpha: f64,
}

impl ELU {
    /// Create a new ELU activation layer.
    ///
    /// # Arguments
    /// * `alpha` - Alpha parameter (default: 1.0)
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<B, S, T> Module<B, S, T> for ELU
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply ELU element-wise: x if x > 0 else alpha * (exp(x) - 1)
        // Sparse tensors are handled efficiently at the storage level
        let input_dense = input.to_dense_generic()?;

        let zero = T::zero();
        let one = T::one();
        let alpha = T::from(self.alpha).unwrap();

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| if x > zero { x } else { alpha * (x.exp() - one) })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // ELU has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: ELU has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: ELU behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "ELU"
    }
}

/// Softmax activation function.
///
/// Applies the function `f(x_i) = exp(x_i) / Σ_j exp(x_j)` along the specified dimension.
/// Uses numerically stable implementation with max subtraction.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Softmax, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let softmax = Softmax::new(-1); // Apply along last dimension
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let output = softmax.forward(&input).unwrap();
/// // Output: [~0.09, ~0.24, ~0.67] (sums to 1.0)
/// ```
#[derive(Debug, Clone)]
pub struct Softmax {
    /// Dimension along which to apply softmax
    pub dim: isize,
}

impl Softmax {
    /// Create a new Softmax activation layer.
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to apply softmax (default: -1 for last dimension)
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl<B, S, T> Module<B, S, T> for Softmax
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply Softmax element-wise with normalization: exp(x) / sum(exp(x))
        // Note: Softmax inherently requires dense computation due to normalization across all elements
        // Sparse tensors are converted to dense for accurate probability computation
        let input_dense = input.to_dense_generic()?;

        // Implement 1D softmax (along last dimension) for numerical stability
        // Multi-dimensional softmax would require more complex indexing logic
        let data = input_dense.as_slice();

        // Find max for numerical stability
        let max_val = data
            .iter()
            .copied()
            .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, b| {
                if a > b {
                    a
                } else {
                    b
                }
            });

        // Compute exp(x - max) and sum
        let exp_data: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: T = exp_data.iter().copied().fold(T::zero(), |a, b| a + b);

        // Normalize
        let result_data: Vec<T> = exp_data.iter().map(|&x| x / sum).collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Softmax has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Softmax has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Softmax behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}

/// LogSoftmax activation function.
///
/// Applies the function `f(x_i) = log(softmax(x_i)) = x_i - log(Σ_j exp(x_j))`.
/// Uses numerically stable log-sum-exp trick.
///
/// # Examples
/// ```rust
/// use coeus_nn::{LogSoftmax, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let log_softmax = LogSoftmax::new(-1); // Apply along last dimension
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let output = log_softmax.forward(&input).unwrap();
/// // Output: log probabilities (negative values)
/// ```
#[derive(Debug, Clone)]
pub struct LogSoftmax {
    /// Dimension along which to apply log-softmax
    pub dim: isize,
}

impl LogSoftmax {
    /// Create a new LogSoftmax activation layer.
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to apply log-softmax (default: -1 for last dimension)
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl<B, S, T> Module<B, S, T> for LogSoftmax
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // LogSoftmax inherently requires dense computation due to normalization across all elements
        // Sparse tensors are converted to dense for accurate probability computation
        let input_dense = input.to_dense_generic()?;

        // LogSoftmax: x_i - max(x) - log(Σ_j exp(x_j - max(x)))
        let data = input_dense.as_slice();

        // Find max for numerical stability
        let max_val = data
            .iter()
            .copied()
            .fold(T::from(f64::NEG_INFINITY).unwrap(), |a, b| {
                if a > b {
                    a
                } else {
                    b
                }
            });

        // Compute log(sum(exp(x - max)))
        let exp_sum: T = data
            .iter()
            .map(|&x| (x - max_val).exp())
            .fold(T::zero(), |a, b| a + b);
        let log_sum_exp = max_val + exp_sum.ln();

        // Compute x - log_sum_exp
        let result_data: Vec<T> = data.iter().map(|&x| x - log_sum_exp).collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // LogSoftmax has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: LogSoftmax has no parameters to zero gradients for
    }

    fn train(&mut self, _mode: bool) {
        // No-op: LogSoftmax behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "LogSoftmax"
    }
}

/// PReLU (Parametric ReLU) activation function.
///
/// Applies the function `f(x) = max(0, x) + weight * min(0, x)` where `weight` is learnable.
///
/// # Examples
/// ```rust
/// use coeus_nn::{PReLU, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1).unwrap(); // 1 channel
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = prelu.forward(&input).unwrap();
/// // Output depends on learned weight parameter
/// ```
#[derive(Debug, Clone)]
pub struct PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Learnable weight parameter (negative slopes) - shape [num_parameters]
    pub weight: Parameter<B, S, T>,
    /// Number of channels (1 for shared weight, or num_channels for per-channel)
    pub num_parameters: usize,
    /// Phantom data to ensure proper generic parameter handling
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new PReLU activation layer.
    ///
    /// # Arguments
    /// * `num_parameters` - Number of weight parameters (1 for shared, or num_channels for per-channel)
    pub fn new(num_parameters: usize) -> Result<Self> {
        if num_parameters == 0 {
            return Err(NNError::InvalidConfiguration {
                message: "num_parameters must be > 0".to_string(),
            });
        }

        // Initialize weight to 0.25 (PyTorch default) - single parameter with shape [num_parameters]
        let weight_data = Tensor::<B, S, T>::from_vec(
            vec![T::from(0.25).unwrap(); num_parameters],
            &[num_parameters],
        )?;

        let weight = Parameter::new(weight_data.requires_grad_(true), "weight".to_string());

        Ok(Self {
            weight,
            num_parameters,
            _phantom: PhantomData,
        })
    }
}

impl<B, S, T> Module<B, S, T> for PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // PReLU: max(0, x) + weight * min(0, x)
        let input_dense = input.to_dense_generic()?;
        let zero = T::zero();
        let weight_data = self.weight.data().as_slice();
        let input_shape = input.shape().dims();

        let result_data: Vec<T> = if input_shape.len() >= 2 {
            // Multi-dimensional tensor: assume [..., C, ...] where C is channels
            // For simplicity, assume channel dimension is 1 (second dimension)
            let channels = input_shape[1];
            let batch_size = input_shape[0];
            let spatial_size = input_shape.iter().skip(2).product::<usize>();

            let mut result_data = Vec::with_capacity(input_dense.len());

            for b in 0..batch_size {
                for c in 0..channels {
                    let weight_idx = if self.num_parameters == 1 {
                        0 // Shared weight across all channels
                    } else {
                        c % self.num_parameters // Per-channel or broadcast
                    };
                    let weight = weight_data[weight_idx];

                    for s in 0..spatial_size {
                        let idx = b * channels * spatial_size + c * spatial_size + s;
                        let x = input_dense.as_slice()[idx];
                        let y = if x > zero { x } else { weight * x };
                        result_data.push(y);
                    }
                }
            }
            result_data
        } else {
            // 1D tensor: use first weight
            let weight = weight_data[0];
            input_dense
                .as_slice()
                .iter()
                .map(|&x| if x > zero { x } else { weight * x })
                .collect()
        };

        let result = Tensor::from_vec(result_data, input.shape().dims())?
            .requires_grad_(input.requires_grad());
        Ok(result)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone()] // PReLU has one learnable weight parameter
    }

    fn modules(&self) -> Vec<&dyn Module<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // No-op: PReLU behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "PReLU"
    }
}

/// Mish activation function.
///
/// Applies the function `f(x) = x * tanh(softplus(x))` where `softplus(x) = ln(1 + e^x)`.
/// Mish is a smooth, non-monotonic activation function with better gradient flow than ReLU.
///
/// # Formula
/// ```text
/// Mish(x) = x * tanh(ln(1 + e^x))
/// ```
///
/// # Examples
/// ```rust
/// use coeus_nn::{Mish, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let mish = Mish::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
///     &[3]
/// ).unwrap();
///
/// let output = mish.forward(&input).unwrap();
/// // Mish is smooth and non-monotonic
/// ```
///
/// # References
/// - Misra (2019): "Mish: A Self Regularized Non-Monotonic Activation Function"
/// - Used in modern CNNs for improved gradient flow
#[derive(Debug, Clone)]
pub struct Mish;

impl Mish {
    /// Create a new Mish activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType + FloatExt + PartialOrd> Module<CpuBackend<T>, DenseStorage<T>, T> for Mish {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        // Mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + e^x)
        let result_data: Vec<T> = input
            .as_slice()
            .iter()
            .map(|&x| {
                let x_f64 = x.to_f64().unwrap();
                // softplus(x) = ln(1 + e^x)
                let softplus = (1.0 + x_f64.exp()).ln();
                // Mish(x) = x * tanh(softplus(x))
                let mish = x_f64 * softplus.tanh();
                T::from(mish).unwrap()
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new() // Mish has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Mish has no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Mish behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Mish"
    }
}

/// Hardsigmoid activation function.
///
/// Applies the function `f(x) = clip((x + 3) / 6, 0, 1)`.
/// Hardsigmoid is an efficient approximation of sigmoid for mobile devices.
///
/// # Formula
/// ```text
/// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
/// ```
///
/// # Examples
/// ```rust
/// use coeus_nn::{Hardsigmoid, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let hardsigmoid = Hardsigmoid::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-3.0), Float32::new(0.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let output = hardsigmoid.forward(&input).unwrap();
/// // Output: [0.0, 0.5, 1.0]
/// ```
///
/// # References
/// - Howard et al. (2019): "Searching for MobileNetV3"
/// - Efficient approximation of sigmoid for mobile devices
#[derive(Debug, Clone)]
pub struct Hardsigmoid;

impl Hardsigmoid {
    /// Create a new Hardsigmoid activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Hardsigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for Hardsigmoid
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply Hardsigmoid element-wise: clip((x + 3) / 6, 0, 1)
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;

        // Hardsigmoid: clip((x + 3) / 6, 0, 1)
        let zero = T::zero();
        let one = T::one();
        let three = T::from(3.0).unwrap();
        let six = T::from(6.0).unwrap();

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| {
                let val = (x + three) / six;
                if val < zero {
                    zero
                } else if val > one {
                    one
                } else {
                    val
                }
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Hardsigmoid has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Hardsigmoid has no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Hardsigmoid behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Hardsigmoid"
    }
}

/// Hardswish activation function.
///
/// Applies the function `f(x) = x * hardsigmoid(x)` where `hardsigmoid(x) = clip((x + 3) / 6, 0, 1)`.
/// Hardswish is an efficient approximation of Swish for mobile devices, used in MobileNetV3.
///
/// # Formula
/// ```text
/// Hardswish(x) = x * clip((x + 3) / 6, 0, 1)
/// ```
///
/// # Examples
/// ```rust
/// use coeus_nn::{Hardswish, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let hardswish = Hardswish::new();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(-3.0), Float32::new(0.0), Float32::new(3.0)],
///     &[3]
/// ).unwrap();
///
/// let output = hardswish.forward(&input).unwrap();
/// // Hardswish is smooth and efficient
/// ```
///
/// # References
/// - Howard et al. (2019): "Searching for MobileNetV3"
/// - Efficient approximation of Swish for mobile devices
#[derive(Debug, Clone)]
pub struct Hardswish;

impl Hardswish {
    /// Create a new Hardswish activation layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Hardswish {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, T> Module<B, S, T> for Hardswish
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Apply Hardswish element-wise: x * hardsigmoid(x)
        // Sparse tensors are handled efficiently at the storage level via ActivationOps trait
        let input_dense = input.to_dense_generic()?;

        // Hardswish: x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
        let zero = T::zero();
        let one = T::one();
        let three = T::from(3.0).unwrap();
        let six = T::from(6.0).unwrap();

        let result_data: Vec<T> = input_dense
            .as_slice()
            .iter()
            .map(|&x| {
                // Compute hardsigmoid(x)
                let hardsigmoid_val = {
                    let val = (x + three) / six;
                    if val < zero {
                        zero
                    } else if val > one {
                        one
                    } else {
                        val
                    }
                };
                // Hardswish: x * hardsigmoid(x)
                x * hardsigmoid_val
            })
            .collect();

        Tensor::from_vec(result_data, input.shape().dims()).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        Vec::new() // Hardswish has no learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: Hardswish has no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Hardswish behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Hardswish"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(-1.0),
                Float32::new(0.0),
                Float32::new(0.5),
                Float32::new(2.0),
            ],
            &[4],
        )
        .unwrap();

        let output = relu.forward(&input).unwrap();

        let expected = [0.0, 0.0, 0.5, 2.0];
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e);
        }
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)],
            &[1],
        )
        .unwrap();

        let output = sigmoid.forward(&input).unwrap();

        let actual = output.as_slice()[0].get();
        assert_relative_eq!(actual, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)],
            &[1],
        )
        .unwrap();

        let output = tanh.forward(&input).unwrap();

        let actual = output.as_slice()[0].get();
        assert_relative_eq!(actual, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0), Float32::new(-1.0)],
            &[3],
        )
        .unwrap();

        let output = gelu.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert_relative_eq!(actual[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(actual[1], 0.841, epsilon = 1e-2);
        assert_relative_eq!(actual[2], -0.159, epsilon = 1e-2);
    }

    #[test]
    fn test_swish_forward() {
        let swish = Swish::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(1.0), Float32::new(-1.0)],
            &[3],
        )
        .unwrap();

        let output = swish.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Swish(0) = 0, Swish(1) ≈ 0.731, Swish(-1) ≈ -0.269
        assert_relative_eq!(actual[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(actual[1], 0.731, epsilon = 1e-2);
        assert_relative_eq!(actual[2], -0.269, epsilon = 1e-2);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let leaky_relu = LeakyReLU::new(0.01);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
            &[3],
        )
        .unwrap();

        let output = leaky_relu.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // LeakyReLU(-1) = -0.01, LeakyReLU(0) = 0, LeakyReLU(1) = 1
        assert_relative_eq!(actual[0], -0.01, epsilon = 1e-5);
        assert_relative_eq!(actual[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(actual[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_elu_forward() {
        let elu = ELU::new(1.0);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
            &[3],
        )
        .unwrap();

        let output = elu.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // ELU(-1) ≈ -0.632, ELU(0) = 0, ELU(1) = 1
        assert_relative_eq!(actual[0], -0.632, epsilon = 1e-2);
        assert_relative_eq!(actual[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(actual[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_forward() {
        let softmax = Softmax::new(-1);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output = softmax.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
        assert_relative_eq!(actual[0], 0.09, epsilon = 1e-2);
        assert_relative_eq!(actual[1], 0.24, epsilon = 1e-2);
        assert_relative_eq!(actual[2], 0.67, epsilon = 1e-2);

        // Sum should be 1.0
        let sum: f32 = actual.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_log_softmax_forward() {
        let log_softmax = LogSoftmax::new(-1);

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output = log_softmax.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // LogSoftmax values should be negative (log of probabilities)
        assert!(actual[0] < 0.0);
        assert!(actual[1] < 0.0);
        assert!(actual[2] < 0.0);

        // exp(LogSoftmax) should equal Softmax
        let softmax = Softmax::new(-1);
        let softmax_output = softmax.forward(&input).unwrap();
        let softmax_actual: Vec<f32> = softmax_output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        for (log_val, softmax_val) in actual.iter().zip(softmax_actual.iter()) {
            assert_relative_eq!(log_val.exp(), *softmax_val, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_prelu_forward() {
        let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1).unwrap();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
            &[3],
        )
        .unwrap();

        let output = prelu.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // PReLU with weight=0.25: PReLU(-1) = -0.25, PReLU(0) = 0, PReLU(1) = 1
        assert_relative_eq!(actual[0], -0.25, epsilon = 1e-5);
        assert_relative_eq!(actual[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(actual[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_prelu_parameters() {
        let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1).unwrap();

        let params = prelu.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name(), "weight");
        assert!(params[0].requires_grad());
    }

    #[test]
    fn test_activation_construction() {
        // Test that activation functions can be constructed
        let _relu = ReLU::new();
        let _sigmoid = Sigmoid::new();
        let _tanh = Tanh::new();
        let _gelu = GELU::new();
        let _swish = Swish::new();
        let _leaky_relu = LeakyReLU::new(0.01);
        let _elu = ELU::new(1.0);
        let _softmax = Softmax::new(-1);
        let _log_softmax = LogSoftmax::new(-1);
        let _prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1).unwrap();
    }

    #[test]
    fn test_gelu_edge_cases() {
        let gelu = GELU::new();

        // Test with large positive value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)],
            &[1],
        )
        .unwrap();
        let output = gelu.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // GELU(10) ≈ 10 (approaches identity for large positive values)
        assert_relative_eq!(actual, 10.0, epsilon = 1e-3);

        // Test with large negative value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-10.0)],
            &[1],
        )
        .unwrap();
        let output = gelu.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // GELU(-10) ≈ 0 (approaches zero for large negative values)
        assert_relative_eq!(actual, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let softmax = Softmax::new(-1);

        // Test with large values (should not overflow)
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1000.0),
                Float32::new(1001.0),
                Float32::new(1002.0),
            ],
            &[3],
        )
        .unwrap();

        let output = softmax.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Should still sum to 1.0 without overflow
        let sum: f32 = actual.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // All values should be valid (not NaN or Inf)
        for val in actual.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mish_forward() {
        let mish = Mish::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
            &[3],
        )
        .unwrap();

        let output = mish.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Mish(-1) ≈ -0.303
        assert_relative_eq!(actual[0], -0.303, epsilon = 0.01);
        // Mish(0) = 0
        assert_relative_eq!(actual[1], 0.0, epsilon = 1e-5);
        // Mish(1) ≈ 0.865
        assert_relative_eq!(actual[2], 0.865, epsilon = 0.01);
    }

    #[test]
    fn test_mish_edge_cases() {
        let mish = Mish::new();

        // Test with large positive value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)],
            &[1],
        )
        .unwrap();
        let output = mish.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // Mish(10) ≈ 10 (approaches identity for large positive values)
        assert_relative_eq!(actual, 10.0, epsilon = 0.01);

        // Test with large negative value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-10.0)],
            &[1],
        )
        .unwrap();
        let output = mish.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // Mish(-10) ≈ 0 (approaches zero for large negative values)
        assert_relative_eq!(actual, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_hardsigmoid_forward() {
        let hardsigmoid = Hardsigmoid::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-3.0), Float32::new(0.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output = hardsigmoid.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Hardsigmoid(-3) = clip((−3 + 3) / 6, 0, 1) = 0
        assert_relative_eq!(actual[0], 0.0, epsilon = 1e-5);
        // Hardsigmoid(0) = clip((0 + 3) / 6, 0, 1) = 0.5
        assert_relative_eq!(actual[1], 0.5, epsilon = 1e-5);
        // Hardsigmoid(3) = clip((3 + 3) / 6, 0, 1) = 1.0
        assert_relative_eq!(actual[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hardsigmoid_clipping() {
        let hardsigmoid = Hardsigmoid::new();

        // Test clipping at lower bound
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-10.0)],
            &[1],
        )
        .unwrap();
        let output = hardsigmoid.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        assert_relative_eq!(actual, 0.0, epsilon = 1e-5);

        // Test clipping at upper bound
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)],
            &[1],
        )
        .unwrap();
        let output = hardsigmoid.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        assert_relative_eq!(actual, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hardswish_forward() {
        let hardswish = Hardswish::new();

        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-3.0), Float32::new(0.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let output = hardswish.forward(&input).unwrap();
        let actual: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Hardswish(-3) = -3 * 0 = 0
        assert_relative_eq!(actual[0], 0.0, epsilon = 1e-5);
        // Hardswish(0) = 0 * 0.5 = 0
        assert_relative_eq!(actual[1], 0.0, epsilon = 1e-5);
        // Hardswish(3) = 3 * 1.0 = 3.0
        assert_relative_eq!(actual[2], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hardswish_edge_cases() {
        let hardswish = Hardswish::new();

        // Test with large positive value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)],
            &[1],
        )
        .unwrap();
        let output = hardswish.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // Hardswish(10) = 10 * 1.0 = 10.0
        assert_relative_eq!(actual, 10.0, epsilon = 1e-5);

        // Test with large negative value
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-10.0)],
            &[1],
        )
        .unwrap();
        let output = hardswish.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // Hardswish(-10) = -10 * 0 = 0
        assert_relative_eq!(actual, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hardswish_intermediate_values() {
        let hardswish = Hardswish::new();

        // Test with x = 1.5
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.5)],
            &[1],
        )
        .unwrap();
        let output = hardswish.forward(&input).unwrap();
        let actual = output.as_slice()[0].get();
        // Hardswish(1.5) = 1.5 * clip((1.5 + 3) / 6, 0, 1) = 1.5 * 0.75 = 1.125
        assert_relative_eq!(actual, 1.125, epsilon = 1e-5);
    }
}

#[cfg(test)]
mod activation_forward_var_tests {
    // Empty test module placeholder
}
