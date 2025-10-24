//! Normalize transform
//!
//! Normalizes tensor data using mean and standard deviation.
//! Commonly used for image preprocessing and feature scaling.

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;

use super::{Transform, TransformError};

/// Transform that normalizes tensor data
///
/// Applies standardization: (x - mean) / std for each channel/feature.
/// Supports per-channel normalization for multi-dimensional data.
pub struct Normalize {
    /// Mean values for each channel/feature
    mean: Vec<f32>,
    /// Standard deviation values for each channel/feature
    std: Vec<f32>,
}

impl Normalize {
    /// Create a new Normalize transform
    ///
    /// # Arguments
    /// * `mean` - Mean values for normalization (one per channel/feature)
    /// * `std` - Standard deviation values for normalization (one per channel/feature)
    ///
    /// # Panics
    /// Panics if mean and std have different lengths
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "Mean and std must have the same length, got {} and {}",
            mean.len(),
            std.len()
        );

        // Validate that std values are positive
        for &s in &std {
            assert!(s > 0.0, "Standard deviation must be positive, got {}", s);
        }

        Self { mean, std }
    }

    /// Create a normalize transform for single-channel data
    ///
    /// # Arguments
    /// * `mean` - Mean value for normalization
    /// * `std` - Standard deviation value for normalization
    pub fn single_channel(mean: f32, std: f32) -> Self {
        assert!(std > 0.0, "Standard deviation must be positive");
        Self {
            mean: vec![mean],
            std: vec![std],
        }
    }

    /// Create a normalize transform for 3-channel RGB images (ImageNet defaults)
    pub fn imagenet() -> Self {
        Self::new(
            vec![0.485, 0.456, 0.406], // ImageNet means
            vec![0.229, 0.224, 0.225], // ImageNet stds
        )
    }

    /// Create a normalize transform for grayscale images (MNIST/CIFAR-10 style)
    pub fn grayscale() -> Self {
        Self::single_channel(0.1307, 0.3081) // MNIST mean/std
    }

    /// Get the mean values
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Get the standard deviation values
    pub fn std(&self) -> &[f32] {
        &self.std
    }

    /// Legacy method for backward compatibility - deprecated
    pub fn apply_tensor(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        // Convert to f64 for precision, then back to original type
        let mean_f64: Vec<f64> = self.mean.iter().map(|&x| x as f64).collect();
        let std_f64: Vec<f64> = self.std.iter().map(|&x| x as f64).collect();

        let shape = input.shape().dims();

        // Handle different tensor shapes
        match shape.len() {
            1 => {
                // 1D tensor - single channel
                if self.mean.len() != 1 {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!(
                            "1D tensor with 1 channel, got {} channels",
                            self.mean.len()
                        ),
                        actual: format!("1D tensor with shape {:?}", shape),
                    });
                }
                self.normalize_generic(input, &mean_f64, &std_f64, shape)
            }
            2 => {
                // 2D tensor - sequence or feature matrix
                if shape[0] != self.mean.len() {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!(
                            "2D tensor with {} features, got {}",
                            self.mean.len(),
                            shape[0]
                        ),
                        actual: format!("2D tensor with shape {:?}", shape),
                    });
                }
                self.normalize_generic(input, &mean_f64, &std_f64, shape)
            }
            3 => {
                // 3D tensor - image-like (C, H, W)
                if shape[0] != self.mean.len() {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!(
                            "3D tensor with {} channels, got {}",
                            self.mean.len(),
                            shape[0]
                        ),
                        actual: format!("3D tensor with shape {:?}", shape),
                    });
                }
                self.normalize_generic(input, &mean_f64, &std_f64, shape)
            }
            4 => {
                // 4D tensor - batch of images (N, C, H, W)
                if shape[1] != self.mean.len() {
                    return Err(TransformError::ShapeMismatch {
                        expected: format!(
                            "4D tensor with {} channels, got {}",
                            self.mean.len(),
                            shape[1]
                        ),
                        actual: format!("4D tensor with shape {:?}", shape),
                    });
                }
                self.normalize_generic(input, &mean_f64, &std_f64, shape)
            }
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }

    fn normalize_generic(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        mean: &[f64],
        std: &[f64],
        shape: &[usize],
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        let slice = input.as_slice();

        // Apply normalization in the appropriate dimension
        let normalized_values: Vec<Float32> = self.normalize_slice(slice, mean, std, shape);

        // Create new tensor with normalized values - always Dense for now
        let normalized_tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                normalized_values,
                shape,
            )?;

        Ok(normalized_tensor)
    }

    fn normalize_slice(
        &self,
        slice: &[Float32],
        mean: &[f64],
        std: &[f64],
        shape: &[usize],
    ) -> Vec<Float32> {
        match shape.len() {
            1 => self.normalize_1d_slice(slice, mean, std),
            2 => self.normalize_2d_slice(slice, mean, std, shape),
            3 => self.normalize_3d_slice(slice, mean, std, shape),
            4 => self.normalize_4d_slice(slice, mean, std, shape),
            _ => unreachable!("Shape validation already performed"),
        }
    }

    fn normalize_1d_slice(&self, slice: &[Float32], mean: &[f64], _std: &[f64]) -> Vec<Float32> {
        slice
            .iter()
            .map(|&val| {
                let val_f64 = val.get() as f64;
                let normalized = (val_f64 - mean[0]) / self.std[0] as f64;
                Float32::new(normalized as f32)
            })
            .collect()
    }

    fn normalize_2d_slice(
        &self,
        slice: &[Float32],
        mean: &[f64],
        std: &[f64],
        shape: &[usize],
    ) -> Vec<Float32> {
        let (_, seq_len) = (shape[0], shape[1]);
        let mut result = Vec::with_capacity(slice.len());

        for i in 0..shape[0] {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                let val_f64 = slice[idx].get() as f64;
                let normalized = (val_f64 - mean[i]) / std[i];
                result.push(Float32::new(normalized as f32));
            }
        }
        result
    }

    fn normalize_3d_slice(
        &self,
        slice: &[Float32],
        mean: &[f64],
        std: &[f64],
        shape: &[usize],
    ) -> Vec<Float32> {
        let (channels, height, width) = (shape[0], shape[1], shape[2]);
        let mut result = Vec::with_capacity(slice.len());

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let idx = c * height * width + h * width + w;
                    let val_f64 = slice[idx].get() as f64;
                    let normalized = (val_f64 - mean[c]) / std[c];
                    result.push(Float32::new(normalized as f32));
                }
            }
        }
        result
    }

    fn normalize_4d_slice(
        &self,
        slice: &[Float32],
        mean: &[f64],
        std: &[f64],
        shape: &[usize],
    ) -> Vec<Float32> {
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let mut result = Vec::with_capacity(slice.len());

        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let idx =
                            b * channels * height * width + c * height * width + h * width + w;
                        let val_f64 = slice[idx].get() as f64;
                        let normalized = (val_f64 - mean[c]) / std[c];
                        result.push(Float32::new(normalized as f32));
                    }
                }
            }
        }
        result
    }
}

impl
    Transform<
        &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    > for Normalize
{
    fn apply(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        self.apply_tensor(input)
    }
}

impl
    Transform<
        Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    > for Normalize
{
    fn apply(
        &self,
        input: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        self.apply_tensor(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_single_channel() {
        let transform = Normalize::single_channel(2.0, 1.0);
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let result = transform.apply(&input).unwrap();
        let slice = result.as_slice();

        // (1-2)/1 = -1, (2-2)/1 = 0, (3-2)/1 = 1
        assert!((slice[0].get() - (-1.0)).abs() < 1e-6);
        assert!((slice[1].get() - 0.0).abs() < 1e-6);
        assert!((slice[2].get() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_3d_image() {
        let transform = Normalize::new(vec![0.5, 1.0], vec![0.5, 0.5]);
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.0),
                Float32::new(1.0),
                Float32::new(1.0),
                Float32::new(2.0),
            ],
            &[2, 1, 2], // 2 channels, 1 height, 2 width
        )
        .unwrap();

        let result = transform.apply(&input).unwrap();
        let slice = result.as_slice();

        // Channel 0: (0-0.5)/0.5 = -1, (1-0.5)/0.5 = 1
        // Channel 1: (1-1.0)/0.5 = 0, (2-1.0)/0.5 = 2
        assert!((slice[0].get() - (-1.0)).abs() < 1e-6);
        assert!((slice[1].get() - 1.0).abs() < 1e-6);
        assert!((slice[2].get() - 0.0).abs() < 1e-6);
        assert!((slice[3].get() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_imagenet() {
        let transform = Normalize::imagenet();
        assert_eq!(transform.mean(), &[0.485, 0.456, 0.406]);
        assert_eq!(transform.std(), &[0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_normalize_grayscale() {
        let transform = Normalize::grayscale();
        assert_eq!(transform.mean(), &[0.1307]);
        assert_eq!(transform.std(), &[0.3081]);
    }

    #[test]
    fn test_normalize_invalid_shape() {
        let transform = Normalize::new(vec![0.5, 1.0], vec![0.5, 0.5]);
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)], // Wrong number of channels
            &[1],
        )
        .unwrap();

        let result = transform.apply(&input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::ShapeMismatch { .. } => {}
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    #[should_panic]
    fn test_normalize_zero_std() {
        let _transform = Normalize::new(vec![0.5], vec![0.0]);
    }

    #[test]
    #[should_panic]
    fn test_normalize_negative_std() {
        let _transform = Normalize::new(vec![0.5], vec![-1.0]);
    }

    #[test]
    #[should_panic]
    fn test_normalize_mismatched_mean_std() {
        let _transform = Normalize::new(vec![0.5], vec![0.5, 1.0]);
    }
}
