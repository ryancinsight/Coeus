use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

use crate::transforms::TransformError;

/// Apply simple box filter for antialiasing (3D)
pub(crate) fn apply_box_filter_3d(
    tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
    channels: usize,
    height: usize,
    width: usize,
    radius: usize,
) -> Result<Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>, TransformError> {
    let slice = tensor.as_slice();
    let mut filtered_data = Vec::with_capacity(slice.len());

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut count = 0;

                // Simple box filter
                for dy in -(radius as isize)..=(radius as isize) {
                    let y_neighbor = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                    for dx in -(radius as isize)..=(radius as isize) {
                        let x_neighbor = (x as isize + dx).clamp(0, width as isize - 1) as usize;

                        let idx = c * height * width + y_neighbor * width + x_neighbor;
                        sum += slice[idx].get();
                        count += 1;
                    }
                }

                filtered_data.push(Float32::new(sum / count as f32));
            }
        }
    }

    Tensor::from_vec(filtered_data, &[channels, height, width]).map_err(TransformError::TensorError)
}
