use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

use crate::transforms::TransformError;

/// Bicubic resize for 3D tensor
pub(crate) fn resize_bicubic_3d(
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
            let y_base = src_y.floor() as isize;

            for x in 0..target_width {
                let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                let x_base = src_x.floor() as isize;

                let mut val = 0.0;

                // Bicubic interpolation using 4x4 neighborhood
                for dy in -1..=2 {
                    let y_idx = y_base + dy;
                    let y_clamped = y_idx.clamp(0, height as isize - 1) as usize;
                    let wy = bicubic_weight(src_y - y_idx as f32);

                    for dx in -1..=2 {
                        let x_idx = x_base + dx;
                        let x_clamped = x_idx.clamp(0, width as isize - 1) as usize;
                        let wx = bicubic_weight(src_x - x_idx as f32);

                        let idx = c * height * width + y_clamped * width + x_clamped;
                        val += slice[idx].get() * wx * wy;
                    }
                }

                resized_data.push(Float32::new(val));
            }
        }
    }

    Tensor::from_vec(resized_data, &[channels, target_height, target_width])
        .map_err(TransformError::TensorError)
}

pub(crate) fn resize_bicubic_4d(
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

        let resized = resize_bicubic_3d(
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

/// Bicubic interpolation weight function
fn bicubic_weight(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        1.0 - 2.0 * abs_x * abs_x + abs_x * abs_x * abs_x
    } else if abs_x < 2.0 {
        4.0 - 8.0 * abs_x + 5.0 * abs_x * abs_x - abs_x * abs_x * abs_x
    } else {
        0.0
    }
}
