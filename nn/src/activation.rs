//! Advanced Activation Functions for Neural Networks
//!
//! Implementation of state-of-the-art activation functions including:
//! - SwiGLU (Swish-Gated Linear Unit)
//! - GeLU variants
//! - Advanced activation compositions

use crate::error::{NNError, Result};
use crate::backend_crate::Backend;
use crate::storage_crate::{Storage, DenseStorage, StorageFromVec, StorageToDense};
use crate::dtype_crate::DataType;
use crate::tensor_crate::FloatExt;
use crate::tensor_crate::Tensor;

/// Swish-Gated Linear Unit (SwiGLU) activation function
///
/// SwiGLU(x, y) = x * σ(y) where σ is the sigmoid function
/// This is used in modern transformer architectures like PaLM and LLaMA
pub struct SwiGLU<B, S, T> {
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
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
            return Err(NNError::InvalidInput(
                "SwiGLU inputs must have the same shape".to_string(),
            ));
        }

        // Compute sigmoid of y: σ(y) = 1 / (1 + exp(-y))
        let sigmoid_y = self.sigmoid(y)?;

        // Compute element-wise multiplication: x * σ(y)
        let result = x.mul(&sigmoid_y)?;

        Ok(result)
    }

    /// Compute sigmoid function: σ(x) = 1 / (1 + exp(-x))
    fn sigmoid(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // sigmoid(x) = 1 / (1 + exp(-x))

        // First compute exp(-x)
        let neg_x = x.neg()?;
        let exp_neg_x = neg_x.exp()?;

        // Then compute 1 + exp(-x)
        let one = Tensor::ones_like(x)?;
        let denominator = one.add(&exp_neg_x)?;

        // Finally compute 1 / (1 + exp(-x))
        let result = one.div(&denominator)?;

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
        let shape = input.shape().dims();
        let last_dim = *shape.last().ok_or_else(|| {
            NNError::InvalidInput("Input tensor must have at least one dimension".to_string())
        })?;

        if last_dim % 2 != 0 {
            return Err(NNError::InvalidInput(
                "Last dimension must be even for automatic gating split".to_string(),
            ));
        }

        let half_dim = last_dim / 2;

        // Split the input into two halves along the last dimension
        let x = input.slice(&[0..shape.len() - 1], &[0..half_dim])?;
        let y = input.slice(&[0..shape.len() - 1], &[half_dim..last_dim])?;

        self.forward(&x, &y)
    }
}

/// GeLU (Gaussian Error Linear Unit) activation function
///
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub struct GeLU<B, S, T> {
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new GeLU activation function
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Apply GeLU activation using the tanh approximation
    ///
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Compute x^3
        let x_cubed = x.mul(&x.mul(x)?)?;

        // Compute 0.044715 * x^3
        let coeff = T::from(0.044715).unwrap();
        let scaled_cubed = x_cubed.mul_scalar(coeff)?;

        // Compute x + 0.044715 * x^3
        let inner_term = x.add(&scaled_cubed)?;

        // Compute sqrt(2/π) ≈ 0.7978845608
        let sqrt_2_pi = T::from(0.7978845608).unwrap();
        let scaled_inner = inner_term.mul_scalar(sqrt_2_pi)?;

        // Compute tanh(scaled_inner)
        let tanh_result = scaled_inner.tanh()?;

        // Compute 1 + tanh_result
        let one = Tensor::ones_like(x)?;
        let tanh_plus_one = one.add(&tanh_result)?;

        // Compute 0.5 * x * (1 + tanh_result)
        let half = T::from(0.5).unwrap();
        let result = x.mul(&tanh_plus_one)?.mul_scalar(half)?;

        Ok(result)
    }
}

/// SiLU (Sigmoid Linear Unit) activation function
///
/// SiLU(x) = x * sigmoid(x)
/// Also known as Swish: Swish(x) = x * sigmoid(x)
pub struct SiLU<B, S, T> {
    swiglu: SwiGLU<B, S, T>,
}

impl<B, S, T> SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create a new SiLU activation function
    pub fn new() -> Self {
        Self {
            swiglu: SwiGLU::new(),
        }
    }

    /// Apply SiLU activation: x * sigmoid(x)
    pub fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // SiLU(x) = x * sigmoid(x), which is equivalent to SwiGLU(x, x)
        self.swiglu.forward(x, x)
    }
}

/// Activation function registry for dynamic activation selection
pub enum ActivationType {
    SwiGLU,
    GeLU,
    SiLU,
    ReLU,
}

pub struct ActivationFactory<B, S, T> {
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> ActivationFactory<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create an activation function by type
    pub fn create(activation_type: ActivationType) -> Box<dyn Activation<B, S, T>> {
        match activation_type {
            ActivationType::SwiGLU => Box::new(SwiGLU::new()),
            ActivationType::GeLU => Box::new(GeLU::new()),
            ActivationType::SiLU => Box::new(SiLU::new()),
            ActivationType::ReLU => Box::new(ReLU::new()),
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

/// ReLU activation for completeness
pub struct ReLU<B, S, T> {
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> ReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B, S, T> Activation<B, S, T> for ReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // ReLU(x) = max(0, x)
        let zero = Tensor::zeros_like(x)?;
        x.maximum(&zero)
    }
}

impl<B, S, T> Activation<B, S, T> for SwiGLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // For trait compatibility, assume split input for SwiGLU
        self.forward_split(x)
    }
}

impl<B, S, T> Activation<B, S, T> for GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(x)
    }
}

impl<B, S, T> Activation<B, S, T> for SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::dtype::float::Float32;
    use crate::storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestDataType = Float32;

    #[test]
    fn test_swiglu_basic() {
        let swiglu = SwiGLU::<TestBackend, TestStorage, TestDataType>::new();

        // Create test tensors
        let x_data = vec![1.0, -1.0, 2.0, -2.0];
        let y_data = vec![0.0, 0.0, 1.0, 1.0];

        let x = Tensor::new_from_vec(x_data, &[2, 2]).unwrap();
        let y = Tensor::new_from_vec(y_data, &[2, 2]).unwrap();

        let result = swiglu.forward(&x, &y).unwrap();

        // Check that result has correct shape
        assert_eq!(result.shape().dims(), &[2, 2]);

        // SwiGLU(1, 0) = 1 * sigmoid(0) = 1 * 0.5 = 0.5
        // SwiGLU(-1, 0) = -1 * sigmoid(0) = -1 * 0.5 = -0.5
        // SwiGLU(2, 1) = 2 * sigmoid(1) ≈ 2 * 0.731 = 1.462
        // SwiGLU(-2, 1) = -2 * sigmoid(1) ≈ -2 * 0.731 = -1.462

        let result_data = result.as_slice();
        assert!(result_data[0] > 0.4 && result_data[0] < 0.6); // ≈ 0.5
        assert!(result_data[1] > -0.6 && result_data[1] < -0.4); // ≈ -0.5
    }

    #[test]
    fn test_gelu_approximation() {
        let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();

        // Test with zero input
        let x_data = vec![0.0];
        let x = Tensor::new_from_vec(x_data, &[1]).unwrap();

        let result = gelu.forward(&x).unwrap();
        let result_data = result.as_slice();

        // GELU(0) should be approximately 0
        assert!(result_data[0] >= -0.1 && result_data[0] <= 0.1);
    }

    #[test]
    fn test_activation_factory() {
        let relu = ActivationFactory::create(ActivationType::ReLU);
        let swiglu = ActivationFactory::create(ActivationType::SwiGLU);

        let x_data = vec![-1.0, 0.0, 1.0];
        let x = Tensor::new_from_vec(x_data, &[3]).unwrap();

        // Test ReLU
        let relu_result = relu.forward(&x).unwrap();
        let relu_data = relu_result.as_slice();
        assert_eq!(relu_data[0], 0.0); // ReLU(-1) = 0
        assert_eq!(relu_data[1], 0.0); // ReLU(0) = 0
        assert_eq!(relu_data[2], 1.0); // ReLU(1) = 1

        // Test SwiGLU (split mode)
        let swiglu_input = vec![1.0, 0.0, -1.0, 1.0, 2.0, -1.0]; // 2 elements per group
        let swiglu_tensor = Tensor::new_from_vec(swiglu_input, &[3, 2]).unwrap();
        let _swiglu_result = swiglu.forward(&swiglu_tensor).unwrap();
        // SwiGLU split test would require more complex assertions
    }
}
