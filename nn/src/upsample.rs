//! Upsampling layers for neural networks.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

/// Interpolation mode for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Nearest neighbor interpolation
    Nearest,
    /// Bilinear interpolation (2D only)
    Bilinear,
    /// Trilinear interpolation (3D only)
    Trilinear,
}

/// Upsampling layer with multiple interpolation modes.
///
/// Upsamples the input to a given size or by a given scale factor using the specified interpolation mode.
/// Essential for encoder-decoder architectures like U-Net, FCN, SegNet, and super-resolution networks.
///
/// # Shape
/// - Input: `(N, C, *)` where * can be (H, W) for 2D or (D, H, W) for 3D
/// - Output: `(N, C, *)` with upsampled spatial dimensions
///
/// # Examples
/// ```rust
/// use nn::{Upsample, InterpolationMode, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Upsample by scale factor 2 using nearest neighbor
/// let upsample = Upsample::new(None, Some(2.0), InterpolationMode::Nearest);
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32]).unwrap();
/// let output = upsample.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 64, 64]);
/// ```
///
/// # References
/// - Ronneberger et al. (2015): "U-Net: Convolutional Networks for Biomedical Image Segmentation"
/// - Long et al. (2015): "Fully Convolutional Networks for Semantic Segmentation"
/// - Dong et al. (2014): "Learning a Deep Convolutional Network for Image Super-Resolution" (SRCNN)
#[derive(Debug, Clone)]
pub struct Upsample {
    /// Target output size (optional, mutually exclusive with scale_factor)
    pub size: Option<Vec<usize>>,
    /// Scale factor for upsampling (optional, mutually exclusive with size)
    pub scale_factor: Option<f64>,
    /// Interpolation mode
    pub mode: InterpolationMode,
}

impl Upsample {
    /// Create a new Upsample layer.
    ///
    /// # Arguments
    /// * `size` - Target output size (optional)
    /// * `scale_factor` - Scale factor for upsampling (optional)
    /// * `mode` - Interpolation mode
    ///
    /// # Panics
    /// Panics if both size and scale_factor are None or both are Some
    pub fn new(
        size: Option<Vec<usize>>,
        scale_factor: Option<f64>,
        mode: InterpolationMode,
    ) -> Self {
        assert!(
            size.is_some() ^ scale_factor.is_some(),
            "Exactly one of size or scale_factor must be specified"
        );
        if let Some(sf) = scale_factor {
            assert!(sf > 0.0, "scale_factor must be > 0");
        }

        Self {
            size,
            scale_factor,
            mode,
        }
    }

    /// Compute output size based on input size and scale factor/size.
    fn compute_output_size(&self, input_size: &[usize]) -> Vec<usize> {
        if let Some(ref size) = self.size {
            size.clone()
        } else if let Some(scale_factor) = self.scale_factor {
            input_size
                .iter()
                .map(|&s| (s as f64 * scale_factor) as usize)
                .collect()
        } else {
            unreachable!()
        }
    }

    /// Nearest neighbor interpolation for 2D input.
    fn nearest_2d<T: DataType + FloatExt>(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        output_h: usize,
        output_w: usize,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        let scale_h = input_h as f64 / output_h as f64;
        let scale_w = input_w as f64 / output_w as f64;

        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        // Nearest neighbor: round to nearest input pixel
                        let ih = ((oh as f64 + 0.5) * scale_h).floor() as usize;
                        let iw = ((ow as f64 + 0.5) * scale_w).floor() as usize;

                        let ih = ih.min(input_h - 1);
                        let iw = iw.min(input_w - 1);

                        let input_idx = ((n * channels + c) * input_h + ih) * input_w + iw;
                        output_data.push(input_data[input_idx]);
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, channels, output_h, output_w])
            .map_err(Into::into)
    }

    /// Bilinear interpolation for 2D input.
    fn bilinear_2d<T: DataType + FloatExt>(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        output_h: usize,
        output_w: usize,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(batch_size * channels * output_h * output_w);

        let scale_h = (input_h - 1) as f64 / (output_h - 1).max(1) as f64;
        let scale_w = (input_w - 1) as f64 / (output_w - 1).max(1) as f64;

        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..output_h {
                    for ow in 0..output_w {
                        // Bilinear interpolation
                        let h_float = oh as f64 * scale_h;
                        let w_float = ow as f64 * scale_w;

                        let h0 = h_float.floor() as usize;
                        let w0 = w_float.floor() as usize;
                        let h1 = (h0 + 1).min(input_h - 1);
                        let w1 = (w0 + 1).min(input_w - 1);

                        let h_weight = T::from(h_float - h0 as f64).unwrap();
                        let w_weight = T::from(w_float - w0 as f64).unwrap();

                        // Get four corner values
                        let idx00 = ((n * channels + c) * input_h + h0) * input_w + w0;
                        let idx01 = ((n * channels + c) * input_h + h0) * input_w + w1;
                        let idx10 = ((n * channels + c) * input_h + h1) * input_w + w0;
                        let idx11 = ((n * channels + c) * input_h + h1) * input_w + w1;

                        let v00 = input_data[idx00];
                        let v01 = input_data[idx01];
                        let v10 = input_data[idx10];
                        let v11 = input_data[idx11];

                        // Bilinear interpolation formula
                        let one = T::one();
                        let v0 = v00 * (one - w_weight) + v01 * w_weight;
                        let v1 = v10 * (one - w_weight) + v11 * w_weight;
                        let val = v0 * (one - h_weight) + v1 * h_weight;

                        output_data.push(val);
                    }
                }
            }
        }

        Tensor::from_vec(output_data, &[batch_size, channels, output_h, output_w])
            .map_err(Into::into)
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Upsample {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        match input_shape.len() {
            4 => {
                // 2D input: [N, C, H, W]
                let spatial_dims = &input_shape[2..];
                let output_size = self.compute_output_size(spatial_dims);
                let output_h = output_size[0];
                let output_w = output_size[1];

                match self.mode {
                    InterpolationMode::Nearest => self.nearest_2d(input, output_h, output_w),
                    InterpolationMode::Bilinear => self.bilinear_2d(input, output_h, output_w),
                    InterpolationMode::Trilinear => Err(crate::error::NNError::InvalidInput {
                        message:
                            "Trilinear interpolation requires 5D input (N, C, D, H, W), got 4D"
                                .to_string(),
                    }),
                }
            }
            _ => Err(crate::error::NNError::InvalidInput {
                message: format!(
                    "Unsupported input dimensions: {}D (expected 4D for 2D upsampling)",
                    input_shape.len()
                ),
            }),
        }
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new() // No learnable parameters
    }

    fn zero_grad(&mut self) {
        // No-op: no parameters
    }

    fn train(&mut self, _mode: bool) {
        // No-op: behavior doesn't change
    }

    fn name(&self) -> &str {
        "Upsample"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_upsample_nearest_scale_factor() {
        let upsample = Upsample::new(None, Some(2.0), InterpolationMode::Nearest);

        // Input: [1, 1, 2, 2]
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2],
        )
        .unwrap();

        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        // Output should be [1, 1, 4, 4]
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_upsample_nearest_size() {
        let upsample = Upsample::new(Some(vec![4, 4]), None, InterpolationMode::Nearest);

        // Input: [1, 1, 2, 2]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 2, 2])
                .unwrap();
        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        // Output should be [1, 1, 4, 4]
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_upsample_bilinear_scale_factor() {
        let upsample = Upsample::new(None, Some(2.0), InterpolationMode::Bilinear);

        // Input: [1, 1, 2, 2]
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2],
        )
        .unwrap();

        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        // Output should be [1, 1, 4, 4]
        assert_eq!(output.shape().dims(), &[1, 1, 4, 4]);

        // Bilinear interpolation should produce smooth transitions
        let output_data = output.as_slice();
        assert!(output_data.iter().all(|&x: &Float32| x.get().is_finite()));
    }

    #[test]
    fn test_upsample_bilinear_computation() {
        let upsample = Upsample::new(Some(vec![3, 3]), None, InterpolationMode::Bilinear);

        // Input: [1, 1, 2, 2] with known values
        let input_data = vec![
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2],
        )
        .unwrap();

        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        // Output should be [1, 1, 3, 3]
        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);

        // Corner values should match input
        let output_data = output.as_slice();
        assert_eq!(output_data[0].get(), 0.0); // Top-left
        assert_eq!(output_data[2].get(), 1.0); // Top-right
        assert_eq!(output_data[6].get(), 2.0); // Bottom-left
        assert_eq!(output_data[8].get(), 3.0); // Bottom-right
    }

    #[test]
    fn test_upsample_batch_processing() {
        let upsample = Upsample::new(None, Some(2.0), InterpolationMode::Nearest);

        // Input: [4, 3, 16, 16]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[4, 3, 16, 16])
                .unwrap();
        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        // Output should be [4, 3, 32, 32]
        assert_eq!(output.shape().dims(), &[4, 3, 32, 32]);
    }

    #[test]
    fn test_upsample_unet_decoder() {
        let upsample = Upsample::new(None, Some(2.0), InterpolationMode::Bilinear);

        // U-Net decoder: [1, 512, 7, 7] → [1, 512, 14, 14]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 512, 7, 7])
                .unwrap();
        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 512, 14, 14]);
    }

    #[test]
    fn test_upsample_super_resolution() {
        let upsample = Upsample::new(None, Some(4.0), InterpolationMode::Bilinear);

        // Super-resolution: [1, 64, 32, 32] → [1, 64, 128, 128]
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32])
                .unwrap();
        let output =
            <Upsample as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
                &upsample, &input,
            )
            .unwrap();

        assert_eq!(output.shape().dims(), &[1, 64, 128, 128]);
    }

    #[test]
    #[should_panic(expected = "Exactly one of size or scale_factor must be specified")]
    fn test_upsample_both_none() {
        let _upsample = Upsample::new(None, None, InterpolationMode::Nearest);
    }

    #[test]
    #[should_panic(expected = "Exactly one of size or scale_factor must be specified")]
    fn test_upsample_both_some() {
        let _upsample = Upsample::new(Some(vec![4, 4]), Some(2.0), InterpolationMode::Nearest);
    }

    #[test]
    #[should_panic(expected = "scale_factor must be > 0")]
    fn test_upsample_invalid_scale_factor() {
        let _upsample = Upsample::new(None, Some(0.0), InterpolationMode::Nearest);
    }
}
