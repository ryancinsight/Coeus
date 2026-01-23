//! Common quantization operations used across different quantization modules

use crate::core::error::{NNError, Result};

use quantization::QuantizationScheme;

use backend::Backend;
use dtype::DataType;
use storage::{QuantizedStorage, Storage, StorageFromVec};
use tensor::Tensor;

/// Common quantization operations that can be shared across different quantization implementations
pub trait QuantizationOps<T> {
    fn to_f64_checked(value: &T) -> Result<f64>
    where
        T: num_traits::ToPrimitive,
    {
        value.to_f64().ok_or_else(|| NNError::NumericalError {
            message: "Failed to convert value to f64".to_string(),
        })
    }

    fn from_f64_checked(value: f64) -> Result<T>
    where
        T: num_traits::FromPrimitive,
    {
        T::from_f64(value).ok_or_else(|| NNError::NumericalError {
            message: "Failed to convert f64 to value".to_string(),
        })
    }

    /// Get the quantization range for a given bitwidth
    fn quantization_range(bits: usize) -> (i32, i32) {
        match bits {
            4 => (-8, 7),          // Signed 4-bit: -8 to 7
            8 => (-128, 127),      // Signed 8-bit: -128 to 127
            16 => (-32768, 32767), // Signed 16-bit
            _ => (0, 0),           // Should not happen
        }
    }

    /// Quantize a single value using the specified scheme
    fn quantize_value(
        val: &T,
        scale: &T,
        zero_point: &T,
        scheme: QuantizationScheme,
        bits: usize,
    ) -> Result<i32>
    where
        T: DataType + PartialOrd + num_traits::ToPrimitive,
    {
        let (qmin, qmax) = Self::quantization_range(bits);

        let val_f = Self::to_f64_checked(val)?;
        let scale_f = Self::to_f64_checked(scale)?;
        let zero_point_f = Self::to_f64_checked(zero_point)?;

        if !scale_f.is_finite() || scale_f == 0.0 {
            return Err(NNError::NumericalError {
                message: "Invalid quantization scale".to_string(),
            });
        }

        let q_f = match scheme {
            QuantizationScheme::Affine => (val_f / scale_f + zero_point_f).round(),
            QuantizationScheme::Symmetric => (val_f / scale_f).round(),
        };

        let q_i = q_f.max(qmin as f64).min(qmax as f64).trunc() as i32;

        Ok(q_i)
    }

    /// Dequantize a single quantized value using the specified scheme
    fn dequantize_value(
        quantized: &i32,
        scale: &T,
        zero_point: &T,
        scheme: QuantizationScheme,
    ) -> Result<T>
    where
        T: DataType + num_traits::ToPrimitive + num_traits::FromPrimitive,
    {
        let scale_f = Self::to_f64_checked(scale)?;
        let zero_point_f = Self::to_f64_checked(zero_point)?;
        if !scale_f.is_finite() {
            return Err(NNError::NumericalError {
                message: "Invalid quantization scale".to_string(),
            });
        }

        let q_f = f64::from(*quantized);
        let value_f = match scheme {
            QuantizationScheme::Affine => (q_f - zero_point_f) * scale_f,
            QuantizationScheme::Symmetric => q_f * scale_f,
        };

        Self::from_f64_checked(value_f)
    }

    /// Create quantized storage from tensor data
    fn create_quantized_storage<const BITS: usize>(
        data: &[T],
        shape: &[usize],
        scale: T,
        zero_point: T,
    ) -> Result<QuantizedStorage<T, BITS>>
    where
        T: DataType
            + core::cmp::PartialOrd
            + num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive,
    {
        QuantizedStorage::<T, BITS>::from_vec_with_params(data, shape, scale, zero_point)
    }

    /// Convert tensor to quantized format
    fn tensor_to_quantized<B, S, const BITS: usize>(
        tensor: &Tensor<B, S, T>,
        scale: T,
        zero_point: T,
    ) -> Result<Tensor<B, QuantizedStorage<T, BITS>, T>>
    where
        B: Backend<Data = T> + Clone,
        S: Storage<T> + Clone + 'static,
        T: DataType
            + core::cmp::PartialOrd
            + num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive,
    {
        let shape = tensor.shape().dims();
        let data = tensor.as_slice();

        let quantized_storage =
            Self::create_quantized_storage::<BITS>(data, shape, scale, zero_point)?;

        Ok(Tensor::<_, QuantizedStorage<T, BITS>, T>::from_storage(
            quantized_storage,
            tensor.backend().clone(),
        ))
    }
}

/// Blanket implementation for any type that implements the required traits
impl<T> QuantizationOps<T> for T where
    T: DataType
        + core::cmp::PartialOrd
        + num_traits::Float
        + num_traits::ToPrimitive
        + num_traits::FromPrimitive
{
}
