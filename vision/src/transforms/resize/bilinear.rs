use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

use super::common::apply_box_filter_3d;
use crate::transforms::TransformError;

/// Bilinear resize for 3D tensor
pub(crate) fn resize_bilinear_3d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    channels: usize,
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let slice = tensor.as_slice();
    let mut resized_data = Vec::with_capacity(channels * target_height * target_width);

    let scale_y = height as f32 / target_height as f32;
    let scale_x = width as f32 / target_width as f32;

    for c in 0..channels {
        for y in 0..target_height {
            let src_y = (y as f32 + 0.5) * scale_y - 0.5;
            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(height - 1);
            let wy = src_y - y0 as f32;

            for x in 0..target_width {
                let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let wx = src_x - x0 as f32;

                // Bilinear interpolation
                let idx00 = c * height * width + y0 * width + x0;
                let idx01 = c * height * width + y0 * width + x1;
                let idx10 = c * height * width + y1 * width + x0;
                let idx11 = c * height * width + y1 * width + x1;

                let val = slice[idx00].get() * (1.0 - wx) * (1.0 - wy)
                    + slice[idx01].get() * wx * (1.0 - wy)
                    + slice[idx10].get() * (1.0 - wx) * wy
                    + slice[idx11].get() * wx * wy;

                resized_data.push(Float32::new(val));
            }
        }
    }

    Tensor::from_vec(resized_data, &[channels, target_height, target_width])
        .map_err(TransformError::TensorError)
}

/// Bilinear resize with antialiasing for 3D tensor
pub(crate) fn resize_bilinear_antialias_3d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    channels: usize,
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    // For antialiasing, apply a simple gaussian-like low-pass filter
    // This is a simplified version - in production, you'd use a proper anti-aliasing kernel
    let sigma = 1.0;
    let radius = (sigma * 2.0) as usize;

    // If downsampling, apply antialiasing
    if target_height < height || target_width < width {
        // Apply simple box filter as approximation
        apply_box_filter_3d(tensor, channels, height, width, radius).and_then(|filtered| {
            resize_bilinear_3d(
                &filtered,
                channels,
                height,
                width,
                target_height,
                target_width,
            )
        })
    } else {
        // No antialiasing needed for upsampling
        resize_bilinear_3d(tensor, channels, height, width, target_height, target_width)
    }
}

pub(crate) fn resize_bilinear_4d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let mut all_batches = Vec::new();

    for b in 0..batch {
        let start_idx = b * channels * height * width;
        let end_idx = (b + 1) * channels * height * width;

        let single_image_data = tensor.as_slice()[start_idx..end_idx].to_vec();
        let single_image = Tensor::from_vec(single_image_data, &[channels, height, width])
            .map_err(TransformError::TensorError)?;

        let resized = resize_bilinear_3d(
            &single_image,
            channels,
            height,
            width,
            target_height,
            target_width,
        )?;
        all_batches.extend_from_slice(resized.as_slice());
    }

    Tensor::from_vec(all_batches, &[batch, channels, target_height, target_width])
        .map_err(TransformError::TensorError)
}

pub(crate) fn resize_bilinear_antialias_4d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let mut all_batches = Vec::new();

    for b in 0..batch {
        let start_idx = b * channels * height * width;
        let end_idx = (b + 1) * channels * height * width;

        let single_image_data = tensor.as_slice()[start_idx..end_idx].to_vec();
        let single_image = Tensor::from_vec(single_image_data, &[channels, height, width])
            .map_err(TransformError::TensorError)?;

        let resized = resize_bilinear_antialias_3d(
            &single_image,
            channels,
            height,
            width,
            target_height,
            target_width,
        )?;
        all_batches.extend_from_slice(resized.as_slice());
    }

    Tensor::from_vec(all_batches, &[batch, channels, target_height, target_width])
        .map_err(TransformError::TensorError)
}
