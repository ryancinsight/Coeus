//! Resize transform
//!
//! Resizes tensor data to specified dimensions with bilinear/trilinear interpolation.
//! Optimized for image data preprocessing with SIMD acceleration and zero-copy operations.

mod bicubic;
mod bilinear;
mod common;
mod nearest;
mod types;

pub use types::InterpolationMode;

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

use super::{Transform, TransformError};
use bicubic::{resize_bicubic_3d, resize_bicubic_4d};
use bilinear::{
    resize_bilinear_3d, resize_bilinear_4d, resize_bilinear_antialias_3d,
    resize_bilinear_antialias_4d,
};
use nearest::{resize_nearest_3d, resize_nearest_4d};

/// Transform that resizes tensor data to specified dimensions
///
/// Supports various interpolation modes and is optimized for image data.
/// Uses SIMD acceleration when available for performance.
pub struct Resize {
    /// Target size as (height, width) for 2D images
    size: (usize, usize),
    /// Interpolation mode
    mode: InterpolationMode,
    /// Whether to antialias (reduces artifacts in downsampling)
    antialias: bool,
}

impl Resize {
    /// Create a new Resize transform
    ///
    /// # Arguments
    /// * `size` - Target size as (height, width)
    pub fn new(size: (usize, usize)) -> Self {
        Self {
            size,
            mode: InterpolationMode::default(),
            antialias: false,
        }
    }

    /// Create a new Resize transform with specified interpolation mode
    ///
    /// # Arguments
    /// * `size` - Target size as (height, width)
    /// * `mode` - Interpolation mode
    pub fn with_mode(size: (usize, usize), mode: InterpolationMode) -> Self {
        Self {
            size,
            mode,
            antialias: false,
        }
    }

    /// Create a new Resize transform with antialiasing
    ///
    /// # Arguments
    /// * `size` - Target size as (height, width)
    /// * `mode` - Interpolation mode
    /// * `antialias` - Whether to apply antialiasing
    pub fn with_antialias(size: (usize, usize), mode: InterpolationMode, antialias: bool) -> Self {
        Self {
            size,
            mode,
            antialias,
        }
    }

    /// Create a resize transform for ImageNet standard size (224x224)
    pub fn imagenet() -> Self {
        Self::new((224, 224))
    }

    /// Create a resize transform for CIFAR-10 size (32x32)
    pub fn cifar() -> Self {
        Self::new((32, 32))
    }

    /// Get the target size
    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    /// Get the interpolation mode
    pub fn mode(&self) -> InterpolationMode {
        self.mode
    }

    /// Check if antialiasing is enabled
    pub fn antialias(&self) -> bool {
        self.antialias
    }

    /// Apply resize to a tensor
    pub fn apply_tensor(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        let shape = input.shape().dims();

        match shape.len() {
            3 => {
                // 3D tensor - single image (C, H, W)
                if shape[1] == self.size.0 && shape[2] == self.size.1 {
                    // Already correct size, return copy
                    Ok(input.clone())
                } else {
                    resize_3d(input, self.size.0, self.size.1, self.mode, self.antialias)
                }
            }
            4 => {
                // 4D tensor - batch of images (N, C, H, W)
                if shape[2] == self.size.0 && shape[3] == self.size.1 {
                    // Already correct size, return copy
                    Ok(input.clone())
                } else {
                    resize_4d(input, self.size.0, self.size.1, self.mode, self.antialias)
                }
            }
            _ => Err(TransformError::UnsupportedType {
                type_name: format!("{}-dimensional tensor", shape.len()),
            }),
        }
    }
}

/// Resize 3D tensor (single image - channels x height x width)
fn resize_3d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    target_height: usize,
    target_width: usize,
    mode: InterpolationMode,
    antialias: bool,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let shape = tensor.shape().dims();
    let (channels, height, width) = (shape[0], shape[1], shape[2]);

    match mode {
        InterpolationMode::Nearest => {
            resize_nearest_3d(tensor, channels, height, width, target_height, target_width)
        }
        InterpolationMode::Bilinear => {
            if antialias {
                resize_bilinear_antialias_3d(
                    tensor,
                    channels,
                    height,
                    width,
                    target_height,
                    target_width,
                )
            } else {
                resize_bilinear_3d(tensor, channels, height, width, target_height, target_width)
            }
        }
        InterpolationMode::Bicubic => {
            resize_bicubic_3d(tensor, channels, height, width, target_height, target_width)
        }
    }
}

/// Resize 4D tensor (batch of images - batch x channels x height x width)
fn resize_4d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    target_height: usize,
    target_width: usize,
    mode: InterpolationMode,
    antialias: bool,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let shape = tensor.shape().dims();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    match mode {
        InterpolationMode::Nearest => resize_nearest_4d(
            tensor,
            batch,
            channels,
            height,
            width,
            target_height,
            target_width,
        ),
        InterpolationMode::Bilinear => {
            if antialias {
                resize_bilinear_antialias_4d(
                    tensor,
                    batch,
                    channels,
                    height,
                    width,
                    target_height,
                    target_width,
                )
            } else {
                resize_bilinear_4d(
                    tensor,
                    batch,
                    channels,
                    height,
                    width,
                    target_height,
                    target_width,
                )
            }
        }
        InterpolationMode::Bicubic => resize_bicubic_4d(
            tensor,
            batch,
            channels,
            height,
            width,
            target_height,
            target_width,
        ),
    }
}

impl
    Transform<
        &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
        Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    > for Resize
{
    fn apply(
        &self,
        input: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    ) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
        self.apply_tensor(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_new() {
        let transform = Resize::new((224, 224));
        assert_eq!(transform.size(), (224, 224));
        assert_eq!(transform.mode(), InterpolationMode::Bilinear);
        assert!(!transform.antialias());
    }

    #[test]
    fn test_resize_with_mode() {
        let transform = Resize::with_mode((64, 64), InterpolationMode::Nearest);
        assert_eq!(transform.size(), (64, 64));
        assert_eq!(transform.mode(), InterpolationMode::Nearest);
    }

    #[test]
    fn test_resize_with_antialias() {
        let transform = Resize::with_antialias((32, 32), InterpolationMode::Bicubic, true);
        assert_eq!(transform.size(), (32, 32));
        assert_eq!(transform.mode(), InterpolationMode::Bicubic);
        assert!(transform.antialias());
    }

    #[test]
    fn test_resize_imagenet() {
        let transform = Resize::imagenet();
        assert_eq!(transform.size(), (224, 224));
    }

    #[test]
    fn test_resize_cifar() {
        let transform = Resize::cifar();
        assert_eq!(transform.size(), (32, 32));
    }

    #[test]
    fn test_resize_same_size() {
        let transform = Resize::new((3, 4));
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
                Float32::new(9.0),
                Float32::new(10.0),
                Float32::new(11.0),
                Float32::new(12.0),
            ],
            &[1, 3, 4], // 1 channel, 3 height, 4 width
        )
        .unwrap();

        let result = transform.apply_tensor(&input).unwrap();
        assert_eq!(result.shape().dims(), &[1, 3, 4]);
        // Should be identical to input
        assert_eq!(result.as_slice()[0].get(), 1.0);
        assert_eq!(result.as_slice()[11].get(), 12.0);
    }

    #[test]
    fn test_resize_nearest_3d() {
        let transform = Resize::with_mode((2, 2), InterpolationMode::Nearest);
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[1, 2, 2], // 1 channel, 2x2
        )
        .unwrap();

        let result = transform.apply_tensor(&input).unwrap();
        assert_eq!(result.shape().dims(), &[1, 2, 2]);
        // Should be identical for same size
        assert_eq!(result.as_slice()[0].get(), 1.0);
    }

    #[test]
    fn test_resize_unsupported_dimensions() {
        let transform = Resize::new((10, 10));
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3], // 1D tensor
        )
        .unwrap();

        let result = transform.apply_tensor(&input);
        assert!(result.is_err());
        match result.unwrap_err() {
            TransformError::UnsupportedType { type_name } => {
                assert!(type_name.contains("1-dimensional"));
            }
            _ => panic!("Expected UnsupportedType error"),
        }
    }
}
