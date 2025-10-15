//! Common quantization operations used across different quantization modules

use crate::error::Result;

use crate::quantization::core::QuantizationScheme;

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageFromVec, QuantizedStorage};
use coeus_tensor::Tensor;

/// Common quantization operations that can be shared across different quantization implementations
pub trait QuantizationOps<T> {
    /// Get the quantization range for a given bitwidth
    const fn quantization_range(bits: usize) -> (i32, i32) {
        match bits {
            4 => (-8, 7),      // Signed 4-bit: -8 to 7
            8 => (-128, 127),  // Signed 8-bit: -128 to 127
            16 => (-32768, 32767), // Signed 16-bit
            _ => (0, 0), // Should not happen
        }
    }

    /// Quantize a single value using the specified scheme
    fn quantize_value(
        val: &T,
        scale: &T,
        zero_point: &T,
        scheme: QuantizationScheme,
        bits: usize,
    ) -> i32
    where
        T: Clone + PartialOrd,
    {
        let (qmin, qmax) = Self::quantization_range(bits);

        match scheme {
            QuantizationScheme::Affine => {
                // Affine: q = round((x - zero_point) / scale)
                if *val >= *zero_point {
                    ((*val - *zero_point) / *scale).round_to_int()
                } else {
                    -((*zero_point - *val) / *scale).round_to_int()
                }
            }
            QuantizationScheme::Symmetric => {
                // Symmetric: q = round(x / scale)
                (*val / *scale).round_to_int()
            }
        }
        .max(qmin)
        .min(qmax)
    }

    /// Dequantize a single quantized value using the specified scheme
    fn dequantize_value(
        quantized: &i32,
        scale: &T,
        zero_point: &T,
        scheme: QuantizationScheme,
    ) -> T
    where
        T: Clone,
    {
        match scheme {
            QuantizationScheme::Affine => {
                // x_dq = (q - zero_point) * scale
                let q_f32 = *quantized as f32;
                let zp_f32 = zero_point.to_f64().unwrap_or(0.0) as f32;
                let scale_f32 = scale.to_f64().unwrap_or(1.0) as f32;
                T::from_f64((q_f32 - zp_f32) * scale_f32).unwrap_or_else(|| *zero_point)
            }
            QuantizationScheme::Symmetric => {
                // x_dq = q * scale
                let q_f32 = *quantized as f32;
                let scale_f32 = scale.to_f64().unwrap_or(1.0) as f32;
                T::from_f64(q_f32 * scale_f32).unwrap_or_else(T::zero)
            }
        }
    }

    /// Create quantized storage from tensor data
    fn create_quantized_storage<const BITS: usize>(
        data: &[T],
        shape: &[usize],
        scale: T,
        zero_point: T,
    ) -> Result<QuantizedStorage<T, BITS>>
    where
        T: Clone,
    {
        QuantizedStorage::<T, BITS>::from_vec(data.to_vec(), shape, scale, zero_point)
    }

    /// Convert tensor to quantized format
    fn tensor_to_quantized<const BITS: usize>(
        tensor: &Tensor<impl Backend, impl Storage<T>, T>,
        scale: T,
        zero_point: T,
    ) -> Result<Tensor<impl Backend, QuantizedStorage<T, BITS>, T>>
    where
        T: Clone,
    {
        let shape = tensor.shape().dims();
        let data = tensor.as_slice();

        let quantized_storage = Self::create_quantized_storage::<BITS>(data, shape, scale, zero_point)?;

        Ok(Tensor::<_, QuantizedStorage<T, BITS>, T>::from_storage(
            quantized_storage,
            tensor.backend().clone(),
        ))
    }
}

/// Blanket implementation for any type that implements the required traits
impl<T> QuantizationOps<T> for T where T: DataType + Clone + PartialOrd + Into<f64> + From<f64> {}
