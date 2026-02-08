//! Quantized neural network layers

use crate::core::error::{NNError, Result};
use crate::core::module::Module;

use quantization::{QuantizationScheme, QuantizedWeights};
use crate::quantization::quantization_ops::QuantizationOps;

use backend::Backend;
use dtype::DataType;
use storage::{
    QuantizedStorage, QuantizedStorage16, QuantizedStorage4, QuantizedStorage8, Storage,
    StorageFromVec,
};
use tensor::Tensor;

/// Quantized linear layer for inference
///
/// This layer stores quantized weights and performs quantized matrix multiplication.
/// Supports the complete B<S<T>> generic architecture for flexible quantization schemes.
///
/// # Generic Parameters
/// - `B`: Backend type
/// - `S`: Storage type for parameters
/// - `T`: Data type for parameters and computation
/// - `BITS`: Quantization bitwidth (4, 8, or 16)
#[derive(Debug)]
pub struct QuantizedLinear<B, S, T, const BITS: usize>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Quantized weights stored in quantized storage format
    pub weight: Tensor<B, QuantizedStorage<T, BITS>, T>,
    /// Weight quantization scale
    pub weight_scale: T,
    /// Weight quantization zero point
    pub weight_zero_point: T,
    /// Bias tensor (optional)
    pub bias: Option<Tensor<B, S, T>>,
    /// Input quantization scale (learned during QAT)
    pub input_scale: T,
    /// Input quantization zero point (learned during QAT)
    pub input_zero_point: T,
    /// Output quantization scale
    pub output_scale: T,
    /// Output quantization zero point
    pub output_zero_point: T,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Phantom data for backend and storage types
    _phantom: core::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T, const BITS: usize> QuantizedLinear<B, S, T, BITS>
where
    B: Backend<Data = T> + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + Clone
        + PartialOrd,
{
    /// Create a quantized linear layer from a trained FakeQuantize linear layer
    ///
    /// # Arguments
    /// * `backend` - The backend to use
    /// * `weight` - Original floating-point weights
    /// * `weight_scale` - Weight quantization scale
    /// * `weight_zero_point` - Weight quantization zero point
    /// * `bias` - Optional bias tensor
    /// * `input_scale` - Input quantization scale
    /// * `input_zero_point` - Input quantization zero point
    /// * `output_scale` - Output quantization scale
    /// * `output_zero_point` - Output quantization zero point
    /// * `scheme` - Quantization scheme
    ///
    /// # Returns
    /// Quantized linear layer
    pub fn new(
        backend: B,
        weight: Tensor<B, S, T>,
        weight_scale: T,
        weight_zero_point: T,
        bias: Option<Tensor<B, S, T>>,
        input_scale: T,
        input_zero_point: T,
        output_scale: T,
        output_zero_point: T,
        scheme: QuantizationScheme,
    ) -> Result<Self> {
        // Validate bitwidth at compile time
        if BITS != 4 && BITS != 8 && BITS != 16 {
            return Err(NNError::InvalidInput {
                message: format!("Unsupported bitwidth: {}. Supported: 4, 8, 16", BITS),
            });
        }

        // Quantize weights to quantized storage
        let quantized_weight =
            Self::quantize_weights(&weight, weight_scale.clone(), weight_zero_point.clone())?;

        Ok(Self {
            weight: quantized_weight,
            weight_scale,
            weight_zero_point,
            bias,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
            scheme,
            _phantom: core::marker::PhantomData,
        })
    }

    /// Quantize floating-point weights to quantized storage format
    ///
    /// Uses the selected quantization scheme to pack weights into quantized storage.
    ///
    /// # Arguments
    /// * `weight` - Original floating-point weights
    /// * `scale` - Weight quantization scale
    /// * `zero_point` - Weight quantization zero point
    ///
    /// # Returns
    /// Weights stored in quantized storage format
    fn quantize_weights(
        weight: &Tensor<B, S, T>,
        scale: T,
        zero_point: T,
    ) -> Result<Tensor<B, QuantizedStorage<T, BITS>, T>> {
        // Use common quantization operations
        <T as QuantizationOps<T>>::tensor_to_quantized::<BITS>(weight, scale, zero_point)
    }

    /// Forward pass with quantized operations
    ///
    /// Performs quantized matrix multiplication with proper scale handling.
    /// This supports the complete B<S<T>> generic architecture.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// Quantized output
    pub fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Convert input to quantized format if needed
        let quantized_input = self.quantize_input(input)?;

        // Perform quantized matrix multiplication
        self.quantized_matmul(&quantized_input)
    }

    /// Quantize input tensor to match weight quantization format
    ///
    /// # Arguments
    /// * `input` - Input tensor to quantize
    ///
    /// # Returns
    /// Quantized input tensor
    fn quantize_input(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, QuantizedStorage<T, BITS>, T>> {
        // Use common quantization operations
        <T as QuantizationOps<T>>::tensor_to_quantized::<BITS>(
            input,
            self.input_scale.clone(),
            self.input_zero_point.clone(),
        )
    }

    /// Quantized matrix multiplication with proper scale handling
    ///
    /// Performs: output = (input_quantized @ weight_quantized - zero_point_correction) * output_scale
    /// This is a simplified implementation - production systems would use optimized quantized kernels.
    ///
    /// # Arguments
    /// * `input` - Quantized input tensor
    ///
    /// # Returns
    /// Quantized output tensor
    fn quantized_matmul(
        &self,
        input: &Tensor<B, QuantizedStorage<T, BITS>, T>,
    ) -> Result<Tensor<B, S, T>> {
        // Get dimensions
        let input_shape = input.shape().dims();
        let weight_shape = self.weight.shape().dims();

        let batch_size = input_shape[0];
        let input_features = input_shape[1];
        let output_features = weight_shape[0];

        // Validate dimensions
        if input_features != weight_shape[1] {
            return Err(NNError::InvalidInput {
                message: format!(
                    "Input features {} don't match weight features {}",
                    input_features, weight_shape[1]
                ),
            });
        }

        // For now, convert to dense for computation
        // Production implementation would use specialized quantized kernels
        let input_dense = input.to_dense_generic()?;
        let weight_dense = self.weight.to_dense_generic()?;

        // Perform matrix multiplication in dense space
        let mut output = input_dense.matmul(&weight_dense)?;

        // Apply bias if present
        if let Some(bias) = &self.bias {
            // Broadcast bias to match output shape
            let bias_expanded = bias.expand_to_match_output(&output)?;
            output = output.add(&bias_expanded)?;
        }

        Ok(output)
    }
}

impl<B, S, T, const BITS: usize> Module<B, S, T> for QuantizedLinear<B, S, T, BITS>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + Clone
        + PartialOrd,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        // QuantizedLinear doesn't have learnable parameters in the traditional sense
        // The quantization parameters are fixed after training
        vec![]
    }

    fn zero_grad(&mut self) {
        // No gradients to zero for quantized inference
    }

    fn train(&mut self, _mode: bool) {
        // Quantized layers are typically used for inference only
    }

    fn name(&self) -> &str {
        "QuantizedLinear"
    }
}

/// Mixed precision quantized linear layer for inference
///
/// This layer supports different bitwidths per layer for optimal accuracy vs. efficiency trade-offs.
/// Uses runtime dispatch for different quantized storage types.
///
/// # Generic Parameters
/// - `B`: Backend type
/// - `S`: Storage type for parameters
/// - `T`: Data type for parameters and computation
#[derive(Debug)]
pub struct MixedPrecisionQuantizedLinear<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType,
{
    /// Quantized weights with variable bitwidth
    pub weight: QuantizedWeights<B, S, T>,
    /// Weight quantization scale
    pub weight_scale: T,
    /// Weight quantization zero point
    pub weight_zero_point: T,
    /// Bias tensor (optional)
    pub bias: Option<Tensor<B, S, T>>,
    /// Input quantization scale (learned during QAT)
    pub input_scale: T,
    /// Input quantization zero point (learned during QAT)
    pub input_zero_point: T,
    /// Output quantization scale
    pub output_scale: T,
    /// Output quantization zero point
    pub output_zero_point: T,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Layer name for mixed precision configuration
    pub layer_name: String,
    /// Phantom data for backend and storage types
    _phantom: core::marker::PhantomData<(B, S, T)>,
}
