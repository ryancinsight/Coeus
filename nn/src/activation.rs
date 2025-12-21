//! Advanced Activation Functions for Neural Networks
//!
//! Implementation of state-of-the-art activation functions including:
//! - SwiGLU (Swish-Gated Linear Unit)
//! - GeLU variants
//! - Advanced activation compositions

use crate::error::{NNError, Result};
use crate::functional_activations::tanh as functional_tanh;
use crate::module::Module;
use backend::Backend;
use storage::{Storage, DenseStorage, StorageFromVec, StorageToDense};
use dtype::DataType;
use tensor::{FloatExt, Tensor, ops::arithmetic::*, ops::creation::*};
use crate::parameter::Parameter;

/// Swish-Gated Linear Unit (SwiGLU) activation function
///
/// SwiGLU(x, y) = x * σ(y) where σ is the sigmoid function
/// This is used in modern transformer architectures like PaLM and LLaMA
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
        let one = Tensor::<B, S, T>::ones_generic(x.shape().dims())?;
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

/// GeLU (Gaussian Error Linear Unit) activation function
///
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[derive(Debug)]
pub struct GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> GeLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Clone,
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
        let x_squared = mul(x, x)?;
        let x_cubed = mul(&x_squared, x)?;

        // Compute 0.044715 * x^3
        let coeff = T::from(0.044715).unwrap();
        let scaled_cubed = scalar_mul(&x_cubed, coeff)?;

        // Compute x + 0.044715 * x^3
        let inner_term = add(x, &scaled_cubed)?;

        // Compute sqrt(2/π) ≈ 0.7978845608
        let sqrt_2_pi = T::from(0.7978845608).unwrap();
        let scaled_inner = scalar_mul(&inner_term, sqrt_2_pi)?;

        // Compute tanh(scaled_inner)
        let scaled_inner_dense = scaled_inner.to_dense_generic()?;
        let tanh_result_dense = functional_tanh(&scaled_inner_dense)?;
        let tanh_result_data = tanh_result_dense.as_slice().to_vec();
        let tanh_result = Tensor::<B, S, T>::from_vec(tanh_result_data, scaled_inner.shape().dims())?;

        // Compute 1 + tanh_result
        let one_data = vec![T::from(1.0).unwrap(); x.shape().dims().iter().product()];
        let one = Tensor::<B, S, T>::from_vec(one_data, x.shape().dims())?;
        let tanh_plus_one = add(&one, &tanh_result)?;

        // Compute 0.5 * x * (1 + tanh_result)
        let half = T::from(0.5).unwrap();
        let x_scaled = mul(x, &tanh_plus_one)?;
        let result = scalar_mul(&x_scaled, half)?;

        Ok(result)
    }
}

/// SiLU (Sigmoid Linear Unit) activation function
///
/// SiLU(x) = x * sigmoid(x)
/// Also known as Swish: Swish(x) = x * sigmoid(x)
pub struct SiLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
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
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
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

/// ReLU activation for completeness
#[derive(Debug)]
pub struct ReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
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
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, x: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // ReLU(x) = max(0, x)
        let zero = Tensor::<B, S, T>::zeros_like(x)?;
        Ok(maximum(x, &zero)?)
    }
}

impl<B, S, T> Module<B, S, T> for ReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        <Self as Activation<B, S, T>>::forward(self, input)
    }

    fn parameters(&self) -> Vec<crate::parameter::Parameter<B, S, T>> {
        Vec::new() // ReLU has no parameters
    }

    fn name(&self) -> &str {
        "ReLU"
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

/// PReLU (Parametric ReLU) activation function
///
/// PReLU(x) = max(0, x) + a * min(0, x)
/// where a is a learnable parameter.
#[derive(Debug)]
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
{
    /// Create a new PReLU activation function
    ///
    /// # Arguments
    /// * `num_parameters` - Number of parameters (1 for shared, or number of channels)
    /// * `init_val` - Initial value for a (default 0.25)
    pub fn new(num_parameters: usize, init_val: Option<T>) -> Self {
        let val = init_val.unwrap_or_else(|| T::from(0.25).unwrap());
        let weight_tensor = Tensor::<B, S, T>::from_vec(
            vec![val; num_parameters],
            &[num_parameters],
        )
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
            let neg_scaled = scalar_mul(&neg, w_scalar)?;
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
                let w_reshaped = Tensor::<B, S, T>::from_vec(
                    w_reshaped_dense.as_slice().to_vec(),
                    &shape,
                )?;
                let neg_scaled = mul(&neg, &w_reshaped)?;
                Ok(add(&pos, &neg_scaled)?)
            } else {
                 return Err(NNError::InvalidInput {
                     message: format!("PReLU parameter size {} does not match input shape {:?}", num_params, input_dims),
                 });
            }
        }
    }
}

impl<B, S, T> Activation<B, S, T> for PReLU<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + num_traits::Num + Copy,
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
        let x_data = vec![Float32::new(1.0), Float32::new(-1.0), Float32::new(2.0), Float32::new(-2.0)];
        let y_data = vec![Float32::new(0.0), Float32::new(0.0), Float32::new(1.0), Float32::new(1.0)];

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
        assert!(result_data[1] > Float32::new(-0.6) && result_data[1] < Float32::new(-0.4)); // ≈ -0.5
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
        let swiglu_input: Vec<Float32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -1.0].into_iter().map(Float32::new).collect(); // 2 elements per group
        let swiglu_tensor = Tensor::<TestBackend, TestStorage, TestDataType>::from_vec(swiglu_input, &[3, 2]).unwrap();
        let _swiglu_result = swiglu.forward(&swiglu_tensor).unwrap();
        // SwiGLU split test would require more complex assertions
    }
}
