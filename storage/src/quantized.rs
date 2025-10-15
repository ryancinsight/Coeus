//! Quantized storage implementation
//!
//! Provides memory-efficient quantized storage with configurable bitwidths.
//! Supports 4-bit, 8-bit, and 16-bit quantization with proper packing.

use crate::{DataType, Result, Shape, Storage, StorageError, StorageFromVec};
use alloc::{format, vec, vec::Vec};

/// Quantized storage with configurable bitwidth.
///
/// Packs multiple quantized values into single bytes/words for memory efficiency.
/// Supports 4-bit, 8-bit, and 16-bit quantization schemes.
///
/// # Examples
///
/// ```
/// use coeus_storage::QuantizedStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create 4-bit quantized storage
/// let storage = QuantizedStorage::<Float32, 4>::zeros(&[2, 3]).unwrap();
/// assert_eq!(storage.shape().dims(), &[2, 3]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedStorage<T: DataType, const BITS: usize> {
    /// Packed quantized data
    data: Vec<u8>,
    /// Original shape before quantization
    shape: Shape,
    /// Strides for accessing elements
    strides: Vec<usize>,
    /// Quantization scale factor
    scale: T,
    /// Quantization zero point
    zero_point: T,
}

impl<T: DataType + core::cmp::PartialOrd, const BITS: usize> QuantizedStorage<T, BITS> {
    /// Values per byte for different bitwidths
    const VALUES_PER_BYTE: usize = 8 / BITS;

    /// Creates quantized storage from a vector with specified shape.
    ///
    /// Uses default quantization parameters (scale=1.0, zero_point=0.0).
    /// For custom quantization parameters, use `from_vec_with_params`.
    ///
    /// # Arguments
    /// * `data` - Input data to quantize
    /// * `shape` - Shape of the tensor
    ///
    /// # Returns
    /// Quantized storage or error
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape or invalid bitwidth.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        Self::from_vec_with_params(data, shape, T::one(), T::zero())
    }

    /// Creates quantized storage from a vector with custom quantization parameters.
    ///
    /// # Arguments
    /// * `data` - Original floating-point data to quantize
    /// * `shape` - Tensor shape
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point
    ///
    /// # Errors
    ///
    /// Returns error if data size doesn't match shape or invalid bitwidth.
    pub fn from_vec_with_params(data: Vec<T>, shape: &[usize], scale: T, zero_point: T) -> Result<Self> {
        // Validate bitwidth
        if BITS != 4 && BITS != 8 && BITS != 16 {
            return Err(StorageError::InvalidShape {
                reason: "Unsupported bitwidth",
            });
        }

        // Validate shape matches data size
        let expected_len = shape.iter().product();
        if data.len() != expected_len {
            return Err(StorageError::ShapeMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }

        // Quantize and pack data
        let packed_data = Self::quantize_and_pack(&data, scale, zero_point)?;

        // Calculate strides (row-major)
        let mut strides = vec![1; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }

        Ok(Self {
            data: packed_data,
            shape: Shape::new(shape)?,
            strides,
            scale,
            zero_point,
        })
    }

    /// Creates zero-initialized quantized storage.
    ///
    /// # Arguments
    /// * `shape` - Tensor shape
    ///
    /// # Errors
    ///
    /// Returns error if invalid bitwidth.
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let total_elements = shape.iter().product();
        let zero_data = vec![T::zero(); total_elements];
        Self::from_vec(zero_data, shape)
    }

    /// Creates one-initialized quantized storage.
    ///
    /// # Arguments
    /// * `shape` - Tensor shape
    ///
    /// # Errors
    ///
    /// Returns error if invalid bitwidth.
    pub fn ones(shape: &[usize]) -> Result<Self> {
        let total_elements = shape.iter().product();
        let one_data = vec![T::one(); total_elements];
        Self::from_vec(one_data, shape)
    }

    /// Quantizes floating-point data and packs into bytes.
    ///
    /// # Arguments
    /// * `data` - Floating-point data to quantize
    /// * `scale` - Quantization scale
    /// * `zero_point` - Quantization zero point
    ///
    /// # Returns
    /// Packed quantized data as bytes
    fn quantize_and_pack(data: &[T], scale: T, zero_point: T) -> Result<Vec<u8>> {
        let mut packed = Vec::new();

        // For each group of values that fit in a byte
        for chunk in data.chunks(Self::VALUES_PER_BYTE) {
            let mut byte = 0u8;

            for (i, &value) in chunk.iter().enumerate() {
                // Quantize: q = round((x - zero_point) / scale)
                let quantized = if value >= zero_point {
                    ((value - zero_point) / scale).to_f64().unwrap_or(0.0).round() as i32
                } else {
                    -((zero_point - value) / scale).to_f64().unwrap_or(0.0).round() as i32
                };

                // Clamp to quantization range
                let (qmin, qmax) = Self::quantization_range();
                let clamped = quantized.max(qmin).min(qmax);

                // Pack into byte based on bitwidth
                match BITS {
                    4 => {
                        let nibble = (clamped as u8) & 0x0F;
                        if i % 2 == 0 {
                            byte = nibble;
                        } else {
                            byte |= nibble << 4;
                            packed.push(byte);
                        }
                    }
                    8 => {
                        packed.push(clamped as u8);
                    }
                    16 => {
                        // For 16-bit, we store as two bytes
                        let value = clamped as u16;
                        packed.push((value & 0xFF) as u8);
                        packed.push((value >> 8) as u8);
                    }
                    _ => unreachable!("Validated at creation"),
                }
            }

            // Handle partial byte for 4-bit quantization
            if BITS == 4 && chunk.len() % 2 == 1 {
                packed.push(byte);
            }
        }

        Ok(packed)
    }

    /// Returns quantization range based on bitwidth and sign.
    ///
    /// # Returns
    /// (min, max) quantization values
    fn quantization_range() -> (i32, i32) {
        match BITS {
            4 => (-8, 7),    // Signed 4-bit: -8 to 7
            8 => (-128, 127), // Signed 8-bit: -128 to 127
            16 => (-32768, 32767), // Signed 16-bit
            _ => (0, 0), // Should not happen
        }
    }

    /// Gets quantization scale factor.
    #[must_use]
    pub const fn scale(&self) -> T {
        self.scale
    }

    /// Gets quantization zero point.
    #[must_use]
    pub const fn zero_point(&self) -> T {
        self.zero_point
    }

    /// Gets the bitwidth of this quantized storage.
    #[must_use]
    pub const fn bits(&self) -> usize {
        BITS
    }
}

impl<T: DataType + core::cmp::PartialOrd, const BITS: usize> StorageFromVec<T> for QuantizedStorage<T, BITS> {
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        Self::from_vec(data, shape)
    }

    fn zeros(shape: &[usize]) -> Result<Self> {
        Self::zeros(shape)
    }

    fn ones(shape: &[usize]) -> Result<Self> {
        Self::ones(shape)
    }
}

impl<T: DataType, const BITS: usize> Storage<T> for QuantizedStorage<T, BITS> {
    fn as_slice(&self) -> &[T] {
        // For quantized storage, we can't directly return T slice
        // This is a limitation - quantized storage needs special handling
        // Return empty slice as quantized values need dequantization
        &[]
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // Quantized storage is read-only for raw access
        &mut []
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn len(&self) -> usize {
        self.shape.dims().iter().product()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_contiguous(&self) -> bool {
        // Quantized storage is always considered contiguous
        // since packing makes stride calculations complex
        true
    }

    fn as_storage_ref(&self) -> &dyn Storage<T> {
        self
    }
}

// Type aliases for common quantized storage types
/// 4-bit quantized storage
pub type QuantizedStorage4<T> = QuantizedStorage<T, 4>;
/// 8-bit quantized storage
pub type QuantizedStorage8<T> = QuantizedStorage<T, 8>;
/// 16-bit quantized storage
pub type QuantizedStorage16<T> = QuantizedStorage<T, 16>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageFromVec;
    use coeus_dtype::float::Float32;

    type TestStorage4 = QuantizedStorage<Float32, 4>;
    type TestStorage8 = QuantizedStorage<Float32, 8>;
    type TestStorage16 = QuantizedStorage<Float32, 16>;

    #[test]
    fn test_quantized_storage_4bit_creation() {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let storage = TestStorage4::from_vec_with_params(data, &[2, 2], Float32::new(1.0), Float32::new(0.0)).unwrap();

        assert_eq!(storage.shape().dims(), &[2, 2]);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.bits(), 4);
        assert_eq!(storage.scale(), Float32::new(1.0));
        assert_eq!(storage.zero_point(), Float32::new(0.0));
    }

    #[test]
    fn test_quantized_storage_8bit_creation() {
        let data = vec![
            Float32::new(-1.0),
            Float32::new(0.0),
            Float32::new(1.0),
        ];
        let storage = TestStorage8::from_vec_with_params(data, &[3], Float32::new(0.1), Float32::new(0.0)).unwrap();

        assert_eq!(storage.shape().dims(), &[3]);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.bits(), 8);
    }

    #[test]
    fn test_quantized_storage_16bit_creation() {
        let data = vec![
            Float32::new(100.0),
            Float32::new(200.0),
        ];
        let storage = TestStorage16::from_vec_with_params(data, &[2], Float32::new(10.0), Float32::new(0.0)).unwrap();

        assert_eq!(storage.shape().dims(), &[2]);
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.bits(), 16);
    }

    #[test]
    fn test_quantized_storage_zeros() {
        let storage = TestStorage8::zeros(&[2, 3]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 3]);
        assert_eq!(storage.len(), 6);
        assert_eq!(storage.scale(), Float32::new(1.0));
        assert_eq!(storage.zero_point(), Float32::new(0.0));
    }

    #[test]
    fn test_quantized_storage_ones() {
        let storage = TestStorage8::ones(&[2, 2]).unwrap();
        assert_eq!(storage.shape().dims(), &[2, 2]);
        assert_eq!(storage.len(), 4);
    }

    #[test]
    fn test_quantized_storage_invalid_bitwidth() {
        let data = vec![Float32::new(1.0)];
        let result = QuantizedStorage::<Float32, 3>::from_vec_with_params(data, &[1], Float32::new(1.0), Float32::new(0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_storage_shape_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let result = TestStorage8::from_vec_with_params(data, &[3], Float32::new(1.0), Float32::new(0.0));
        assert!(result.is_err());
    }
}
