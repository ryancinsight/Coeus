//! Quantized storage implementation
//!
//! Provides memory-efficient quantized storage with configurable bitwidths.
//! Supports 4-bit, 8-bit, and 16-bit quantization with proper packing.

use crate::{DataType, Result, Shape, Storage, StorageError, StorageFromVec};
use alloc::{vec, vec::Vec};

/// Quantized storage with configurable bitwidth.
///
/// Packs multiple quantized values into single bytes/words for memory efficiency.
/// Supports 4-bit, 8-bit, and 16-bit quantization schemes.
///
/// # Examples
///
/// ```
/// use storage::{QuantizedStorage, Storage};
/// use dtype::float::Float32;
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

impl<
        T: DataType
            + core::cmp::PartialOrd
            + num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive,
        const BITS: usize,
    > QuantizedStorage<T, BITS>
{
    /// Creates quantized storage from a vector with specified shape.
    ///
    /// Uses default quantization parameters (scale=1.0, `zero_point=0.0`).
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
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        Self::from_vec_with_params(&data, shape, T::one(), T::zero())
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
    pub fn from_vec_with_params(
        data: &[T],
        shape: &[usize],
        scale: T,
        zero_point: T,
    ) -> Result<Self> {
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
        let packed_data = Self::quantize_and_pack(data, scale, zero_point)?;

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
        let num_elements = data.len();
        let max_value = (1 << BITS) - 1; // e.g., 15 for 4-bit

        // Quantize: q = round(x / scale + zero_point)
        let quantized: Vec<usize> = data
            .iter()
            .map(|&x| {
                let scaled = x / scale + zero_point;
                let max_val = T::from_usize(max_value).unwrap_or(T::zero());
                let clamped = scaled.max(T::zero()).min(max_val);
                clamped.to_usize().unwrap_or(0)
            })
            .collect();

        // Pack based on bitwidth
        match BITS {
            4 => {
                // Pack 2 values per byte
                let packed_len = (num_elements + 1) / 2;
                let mut packed = vec![0u8; packed_len];
                for (i, &quantized_val) in quantized.iter().enumerate().take(num_elements) {
                    let byte_idx = i / 2;
                    let is_high_nibble = i % 2 == 0;
                    if is_high_nibble {
                        packed[byte_idx] |= u8::try_from(quantized_val).unwrap_or(0) << 4;
                    } else {
                        packed[byte_idx] |= u8::try_from(quantized_val).unwrap_or(0);
                    }
                }
                Ok(packed)
            }
            8 => {
                // Direct byte mapping
                Ok(quantized
                    .iter()
                    .map(|&q| u8::try_from(q).unwrap_or(0))
                    .collect())
            }
            16 => {
                // Pack 2 bytes per value
                let mut packed = Vec::with_capacity(num_elements * 2);
                for &q in &quantized {
                    packed.push(u8::try_from(q & 0xFF).unwrap_or(0));
                    packed.push(u8::try_from((q >> 8) & 0xFF).unwrap_or(0));
                }
                Ok(packed)
            }
            _ => Err(StorageError::InvalidShape {
                reason: "Unsupported bitwidth",
            }),
        }
    }

    /// Unpacks and dequantizes data
    /// # Errors
    /// Returns error if unpacking fails due to invalid data format
    pub fn unpack_and_dequantize(&self) -> Result<Vec<T>> {
        let num_elements = self.len();
        let mut values = Vec::with_capacity(num_elements);

        match BITS {
            4 => {
                for i in 0..num_elements {
                    let byte_idx = i / 2;
                    let is_high_nibble = i % 2 == 0;
                    let quantized = if is_high_nibble {
                        (self.data[byte_idx] >> 4) & 0x0F
                    } else {
                        self.data[byte_idx] & 0x0F
                    };
                    let dequantized = (T::from_usize(quantized as usize).unwrap_or(T::zero())
                        - self.zero_point)
                        * self.scale;
                    values.push(dequantized);
                }
            }
            8 => {
                for &byte in &self.data {
                    let dequantized = (T::from_usize(byte as usize).unwrap_or(T::zero())
                        - self.zero_point)
                        * self.scale;
                    values.push(dequantized);
                }
            }
            16 => {
                for i in 0..num_elements {
                    let low = self.data[i * 2] as usize;
                    let high = self.data[i * 2 + 1] as usize;
                    let quantized = low | (high << 8);
                    let dequantized = (T::from_usize(quantized).unwrap_or(T::zero())
                        - self.zero_point)
                        * self.scale;
                    values.push(dequantized);
                }
            }
            _ => {
                return Err(StorageError::InvalidShape {
                    reason: "Unsupported bitwidth",
                })
            }
        }

        Ok(values)
    }

    /// Get dequantized element at index
    /// # Errors
    /// Returns error if index is out of bounds
    pub fn get(&self, index: usize) -> Result<T> {
        if index >= self.len() {
            return Err(StorageError::IndexOutOfBounds {
                index,
                bound: self.len(),
            });
        }

        let quantized = match BITS {
            4 => {
                let byte_idx = index / 2;
                let is_high_nibble = index % 2 == 0;
                if is_high_nibble {
                    (self.data[byte_idx] >> 4) & 0x0F
                } else {
                    self.data[byte_idx] & 0x0F
                }
            }
            8 => self.data[index],
            16 => {
                let low = u16::from(self.data[index * 2]);
                let high = u16::from(self.data[index * 2 + 1]);
                u8::try_from((high << 8) | low).unwrap_or(0)
            }
            _ => {
                return Err(StorageError::InvalidShape {
                    reason: "Unsupported bitwidth",
                })
            }
        };

        Ok((T::from_usize(quantized as usize).unwrap_or(T::zero()) - self.zero_point) * self.scale)
    }

    /// Returns quantization range based on bitwidth and sign.
    ///
    /// # Returns
    /// `(min, max)` quantization values
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

impl<
        T: DataType
            + core::cmp::PartialOrd
            + num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive,
        const BITS: usize,
    > StorageFromVec<T> for QuantizedStorage<T, BITS>
{
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        QuantizedStorage::from_vec(data, shape)
    }

    fn zeros(shape: &[usize]) -> Result<Self> {
        QuantizedStorage::zeros(shape)
    }

    fn ones(shape: &[usize]) -> Result<Self> {
        QuantizedStorage::ones(shape)
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

    fn as_storage_ref(&self) -> &Self {
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

impl<T, const BITS: usize> crate::StorageToDense<T> for QuantizedStorage<T, BITS>
where
    T: crate::DataType
        + core::cmp::PartialOrd
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive,
{
    fn to_dense(&self) -> crate::Result<crate::DenseStorage<T>> {
        // Dequantize all values to dense storage
        let mut dense_data = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let quantized_val = self.get(i)?;
            dense_data.push(quantized_val);
        }

        crate::DenseStorage::from_vec(dense_data, self.shape.dims())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

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
        let storage = TestStorage4::from_vec_with_params(
            &data,
            &[2, 2],
            Float32::new(1.0),
            Float32::new(0.0),
        )
        .unwrap();

        assert_eq!(storage.shape().dims(), &[2, 2]);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.bits(), 4);
        assert_eq!(storage.scale(), Float32::new(1.0));
        assert_eq!(storage.zero_point(), Float32::new(0.0));
    }

    #[test]
    fn test_quantized_storage_8bit_creation() {
        let data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)];
        let storage =
            TestStorage8::from_vec_with_params(&data, &[3], Float32::new(0.1), Float32::new(0.0))
                .unwrap();

        assert_eq!(storage.shape().dims(), &[3]);
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.bits(), 8);
    }

    #[test]
    fn test_quantized_storage_16bit_creation() {
        let data = vec![Float32::new(100.0), Float32::new(200.0)];
        let storage =
            TestStorage16::from_vec_with_params(&data, &[2], Float32::new(10.0), Float32::new(0.0))
                .unwrap();

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
        let result = QuantizedStorage::<Float32, 3>::from_vec_with_params(
            &data,
            &[1],
            Float32::new(1.0),
            Float32::new(0.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_storage_shape_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let result =
            TestStorage8::from_vec_with_params(&data, &[3], Float32::new(1.0), Float32::new(0.0));
        assert!(result.is_err());
    }
}
