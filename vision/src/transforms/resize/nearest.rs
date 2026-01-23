use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

use crate::transforms::TransformError;

/// Nearest neighbor resize for 3D tensor
pub(crate) fn resize_nearest_3d(
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
            let src_y = ((y as f32 + 0.5) * scale_y - 0.5).round() as usize;
            let src_y = src_y.min(height - 1);

            for x in 0..target_width {
                let src_x = ((x as f32 + 0.5) * scale_x - 0.5).round() as usize;
                let src_x = src_x.min(width - 1);

                let src_idx = c * height * width + src_y * width + src_x;
                resized_data.push(slice[src_idx]);
            }
        }
    }

    Tensor::from_vec(resized_data, &[channels, target_height, target_width])
        .map_err(TransformError::TensorError)
}

pub(crate) fn resize_nearest_4d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let mut all_batches = Vec::new();

    // Process each sample in the batch separately
    for b in 0..batch {
        let start_idx = b * channels * height * width;
        let end_idx = (b + 1) * channels * height * width;

        // Extract single image tensor
        let single_image_data = tensor.as_slice()[start_idx..end_idx].to_vec();
        let single_image = Tensor::from_vec(single_image_data, &[channels, height, width])
            .map_err(TransformError::TensorError)?;

        // Resize it
        let resized = resize_nearest_3d(
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
