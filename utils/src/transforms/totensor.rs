//! ToTensor transform
//!
//! Converts various data formats (vectors, arrays, etc.) into Coeus tensors.
//! This is typically the first transformation in a preprocessing pipeline.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use super::TransformError;

/// Transform that converts data to tensors
///
/// Converts vectors, arrays, and other data formats into Coeus tensors.
/// Supports f32 and f64 numeric types.
pub struct ToTensor {
    _private: (), // Prevent direct construction
}

impl ToTensor {
    /// Create a new ToTensor transform
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for ToTensor {
    fn default() -> Self {
        Self::new()
    }
}

impl ToTensor {
    /// Apply transform to f32 vector
    pub fn apply_f32(
        &self,
        input: Vec<f32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        let len = input.len();
        // Convert f32 values to Float32 dtype
        let data: Vec<Float32> = input.into_iter().map(Float32::new).collect();

        // Create tensor with shape [length]
        Tensor::from_vec(data, &[len]).map_err(TransformError::TensorError)
    }

    /// Apply transform to f64 vector
    pub fn apply_f64(
        &self,
        input: Vec<f64>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        let len = input.len();
        // Convert f64 to f32, then to Float32 dtype
        let data: Vec<Float32> = input.into_iter().map(|x| Float32::new(x as f32)).collect();

        Tensor::from_vec(data, &[len]).map_err(TransformError::TensorError)
    }

    /// Apply transform to 2D f32 vector
    pub fn apply_f32_2d(
        &self,
        input: Vec<Vec<f32>>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        // Validate that all inner vectors have the same length
        if input.is_empty() {
            return Err(TransformError::InvalidInput {
                message: "Input vector is empty".to_string(),
            });
        }

        let first_len = input[0].len();
        if !input.iter().all(|v| v.len() == first_len) {
            return Err(TransformError::InvalidInput {
                message: "All inner vectors must have the same length".to_string(),
            });
        }

        // Flatten the 2D vector into 1D
        let flat_data: Vec<Float32> = input.into_iter().flatten().map(Float32::new).collect();

        // Create tensor with shape [rows, cols]
        let rows = flat_data.len() / first_len;
        Tensor::from_vec(flat_data, &[rows, first_len]).map_err(TransformError::TensorError)
    }

    /// Apply transform to u8 vector
    pub fn apply_u8(
        &self,
        input: Vec<u8>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        let len = input.len();
        // Convert u8 to f32, then to Float32 dtype (normalize to [0, 1])
        let data: Vec<Float32> = input
            .into_iter()
            .map(|x| Float32::new(x as f32 / 255.0))
            .collect();

        Tensor::from_vec(data, &[len]).map_err(TransformError::TensorError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_totensor_f32_vec() {
        let transform = ToTensor::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = transform.apply_f32(input).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        assert_eq!(result.as_slice()[0].get(), 1.0);
        assert_eq!(result.as_slice()[1].get(), 2.0);
        assert_eq!(result.as_slice()[2].get(), 3.0);
        assert_eq!(result.as_slice()[3].get(), 4.0);
    }

    #[test]
    fn test_totensor_f64_vec() {
        let transform = ToTensor::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = transform.apply_f64(input).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        assert_eq!(result.as_slice()[0].get(), 1.0);
        assert_eq!(result.as_slice()[1].get(), 2.0);
    }

    #[test]
    fn test_totensor_2d_vec() {
        let transform = ToTensor::new();
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let result = transform.apply_f32_2d(input).unwrap();
        assert_eq!(result.shape().dims(), &[3, 2]);
        let slice = result.as_slice();
        assert_eq!(slice[0].get(), 1.0);
        assert_eq!(slice[1].get(), 2.0);
        assert_eq!(slice[2].get(), 3.0);
        assert_eq!(slice[3].get(), 4.0);
        assert_eq!(slice[4].get(), 5.0);
        assert_eq!(slice[5].get(), 6.0);
    }

    #[test]
    fn test_totensor_u8_vec() {
        let transform = ToTensor::new();
        let input = vec![0, 128, 255];

        let result = transform.apply_u8(input).unwrap();
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.as_slice()[0].get(), 0.0);
        assert_eq!(result.as_slice()[1].get(), 128.0 / 255.0);
        assert_eq!(result.as_slice()[2].get(), 1.0);
    }

    #[test]
    fn test_totensor_empty_vec() {
        let transform = ToTensor::new();
        let input: Vec<f32> = vec![];

        let result = transform.apply_f32(input);
        assert!(result.is_ok()); // Empty tensor is valid
        assert_eq!(result.unwrap().shape().dims(), &[0]);
    }

    #[test]
    fn test_totensor_irregular_2d_vec() {
        let transform = ToTensor::new();
        let input = vec![
            vec![1.0, 2.0],
            vec![3.0], // Different length
        ];

        let result = transform.apply_f32_2d(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::InvalidInput { message } => {
                assert!(message.contains("same length"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }
}
