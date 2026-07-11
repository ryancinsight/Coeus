//! Coordinate-grid interpolation operations.

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_tensor::Tensor;

/// Gradients produced by trilinear interpolation reverse mode.
pub struct TrilinearGradients<B: Backend> {
    /// Gradient with respect to image voxel values.
    pub image: Tensor<f32, B>,
    /// Gradient with respect to `(z, y, x)` sampling coordinates.
    pub grid: Tensor<f32, B>,
}

/// Contract failures for [`trilinear_interpolation`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InterpolationError {
    /// The image does not have `[batch, channel, depth, height, width]` shape.
    #[error("trilinear interpolation requires a rank-5 image, got rank {0}")]
    ImageRank(usize),
    /// The grid does not have `[batch, 3, depth, height, width]` shape.
    #[error("trilinear interpolation requires a rank-5 grid, got rank {0}")]
    GridRank(usize),
    /// The grid batch does not match the image batch.
    #[error("trilinear interpolation batch mismatch: image {image}, grid {grid}")]
    BatchMismatch {
        /// Image batch extent.
        image: usize,
        /// Grid batch extent.
        grid: usize,
    },
    /// The grid coordinate axis is not `(z, y, x)`.
    #[error("trilinear interpolation grid must have 3 coordinate channels, got {0}")]
    GridChannels(usize),
    /// An input spatial axis is empty.
    #[error("trilinear interpolation image spatial axis {axis} is empty")]
    EmptySpatialAxis {
        /// Name of the empty image axis.
        axis: &'static str,
    },
    /// An output element count overflowed `usize`.
    #[error("trilinear interpolation output element count overflow")]
    OutputSizeOverflow,
    /// The upstream gradient shape does not match the interpolation output.
    #[error(
        "trilinear interpolation gradient shape mismatch: expected {expected:?}, got {actual:?}"
    )]
    GradientShape {
        /// Shape implied by the image and sampling grid.
        expected: Vec<usize>,
        /// Supplied upstream-gradient shape.
        actual: Vec<usize>,
    },
}

/// Backpropagate through [`trilinear_interpolation`].
///
/// Returns gradients for `(image, grid)`. Image gradients scatter each output
/// contribution to its eight neighbours. Grid gradients differentiate the
/// native-precision trilinear polynomial with respect to `(z, y, x)`; border
/// replication therefore has zero coordinate derivative when both neighbours
/// clamp to the same voxel.
///
/// # Errors
///
/// Returns [`InterpolationError`] under the forward contract failures or when
/// `grad_output` does not have the implied output shape.
pub fn trilinear_interpolation_backward<B>(
    image: &Tensor<f32, B>,
    grid: &Tensor<f32, B>,
    grad_output: &Tensor<f32, B>,
) -> Result<TrilinearGradients<B>, InterpolationError>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let image_shape = image.shape();
    let grid_shape = grid.shape();
    if image_shape.len() != 5 {
        return Err(InterpolationError::ImageRank(image_shape.len()));
    }
    if grid_shape.len() != 5 {
        return Err(InterpolationError::GridRank(grid_shape.len()));
    }
    if image_shape[0] != grid_shape[0] {
        return Err(InterpolationError::BatchMismatch {
            image: image_shape[0],
            grid: grid_shape[0],
        });
    }
    if grid_shape[1] != 3 {
        return Err(InterpolationError::GridChannels(grid_shape[1]));
    }
    for (axis, extent) in [
        ("depth", image_shape[2]),
        ("height", image_shape[3]),
        ("width", image_shape[4]),
    ] {
        if extent == 0 {
            return Err(InterpolationError::EmptySpatialAxis { axis });
        }
    }

    let [batch, channels, depth, height, width] = image_shape.try_into().expect("rank checked");
    let [_, _, out_depth, out_height, out_width] = grid_shape.try_into().expect("rank checked");
    let expected = vec![batch, channels, out_depth, out_height, out_width];
    if grad_output.shape() != expected {
        return Err(InterpolationError::GradientShape {
            expected,
            actual: grad_output.shape().to_vec(),
        });
    }

    let image = image.to_contiguous();
    let grid = grid.to_contiguous();
    let grad_output = grad_output.to_contiguous();
    let image_values = image.as_slice();
    let grid_values = grid.as_slice();
    let upstream = grad_output.as_slice();
    let output_points = out_depth
        .checked_mul(out_height)
        .and_then(|n| n.checked_mul(out_width))
        .ok_or(InterpolationError::OutputSizeOverflow)?;
    let input_points = depth * height * width;
    let mut image_gradient = vec![0.0; image_values.len()];
    let mut grid_gradient = vec![0.0; grid_values.len()];

    for b in 0..batch {
        for point in 0..output_points {
            let coordinate = |axis| grid_values[(b * 3 + axis) * output_points + point];
            let neighbours = |value: f32, extent: usize| {
                let lower_value = value.floor();
                let weight = value - lower_value;
                let lower = lower_value.clamp(0.0, (extent - 1) as f32) as usize;
                let upper = (lower_value + 1.0).clamp(0.0, (extent - 1) as f32) as usize;
                (lower, upper, 1.0 - weight, weight)
            };
            let (z0, z1, wz0, wz1) = neighbours(coordinate(0), depth);
            let (y0, y1, wy0, wy1) = neighbours(coordinate(1), height);
            let (x0, x1, wx0, wx1) = neighbours(coordinate(2), width);
            let corners = [
                (z0, y0, x0, wz0 * wy0 * wx0),
                (z0, y0, x1, wz0 * wy0 * wx1),
                (z0, y1, x0, wz0 * wy1 * wx0),
                (z0, y1, x1, wz0 * wy1 * wx1),
                (z1, y0, x0, wz1 * wy0 * wx0),
                (z1, y0, x1, wz1 * wy0 * wx1),
                (z1, y1, x0, wz1 * wy1 * wx0),
                (z1, y1, x1, wz1 * wy1 * wx1),
            ];

            let mut dz = 0.0;
            let mut dy = 0.0;
            let mut dx = 0.0;
            for channel in 0..channels {
                let base = (b * channels + channel) * input_points;
                let at = |z, y, x| image_values[base + (z * height + y) * width + x];
                let output_index = (b * channels + channel) * output_points + point;
                let grad = upstream[output_index];
                for &(z, y, x, weight) in &corners {
                    image_gradient[base + (z * height + y) * width + x] += grad * weight;
                }

                let x00 = at(z0, y0, x1) - at(z0, y0, x0);
                let x01 = at(z0, y1, x1) - at(z0, y1, x0);
                let x10 = at(z1, y0, x1) - at(z1, y0, x0);
                let x11 = at(z1, y1, x1) - at(z1, y1, x0);
                dx += grad * ((x00 * wy0 + x01 * wy1) * wz0 + (x10 * wy0 + x11 * wy1) * wz1);

                let y0_delta = (at(z0, y1, x0) - at(z0, y0, x0)) * wx0
                    + (at(z0, y1, x1) - at(z0, y0, x1)) * wx1;
                let y1_delta = (at(z1, y1, x0) - at(z1, y0, x0)) * wx0
                    + (at(z1, y1, x1) - at(z1, y0, x1)) * wx1;
                dy += grad * (y0_delta * wz0 + y1_delta * wz1);

                let z_delta_0 = (at(z1, y0, x0) - at(z0, y0, x0)) * wx0
                    + (at(z1, y0, x1) - at(z0, y0, x1)) * wx1;
                let z_delta_1 = (at(z1, y1, x0) - at(z0, y1, x0)) * wx0
                    + (at(z1, y1, x1) - at(z0, y1, x1)) * wx1;
                dz += grad * (z_delta_0 * wy0 + z_delta_1 * wy1);
            }
            grid_gradient[b * 3 * output_points + point] = dz;
            grid_gradient[(b * 3 + 1) * output_points + point] = dy;
            grid_gradient[(b * 3 + 2) * output_points + point] = dx;
        }
    }

    let backend = B::default();
    Ok(TrilinearGradients {
        image: Tensor::from_slice_on(
            [batch, channels, depth, height, width],
            &image_gradient,
            &backend,
        ),
        grid: Tensor::from_slice_on(
            [batch, 3, out_depth, out_height, out_width],
            &grid_gradient,
            &backend,
        ),
    })
}

/// Sample a 3-D image at a voxel-coordinate grid using trilinear interpolation.
///
/// `image` has shape `[batch, channel, depth, height, width]`. `grid` has shape
/// `[batch, 3, output_depth, output_height, output_width]`; its coordinate
/// channels are ordered `(z, y, x)`. Coordinates outside the input extent use
/// border replication. Arithmetic and accumulation use the public `f32`
/// coordinate contract without hidden precision widening.
///
/// # Errors
///
/// Returns [`InterpolationError`] when ranks, batches, coordinate channels, or
/// spatial extents violate the contract, or when the output size overflows.
pub fn trilinear_interpolation<B>(
    image: &Tensor<f32, B>,
    grid: &Tensor<f32, B>,
) -> Result<Tensor<f32, B>, InterpolationError>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let image_shape = image.shape();
    let grid_shape = grid.shape();
    if image_shape.len() != 5 {
        return Err(InterpolationError::ImageRank(image_shape.len()));
    }
    if grid_shape.len() != 5 {
        return Err(InterpolationError::GridRank(grid_shape.len()));
    }
    if image_shape[0] != grid_shape[0] {
        return Err(InterpolationError::BatchMismatch {
            image: image_shape[0],
            grid: grid_shape[0],
        });
    }
    if grid_shape[1] != 3 {
        return Err(InterpolationError::GridChannels(grid_shape[1]));
    }
    for (axis, extent) in [
        ("depth", image_shape[2]),
        ("height", image_shape[3]),
        ("width", image_shape[4]),
    ] {
        if extent == 0 {
            return Err(InterpolationError::EmptySpatialAxis { axis });
        }
    }

    let [batch, channels, depth, height, width] = image_shape.try_into().expect("rank checked");
    let [_, _, out_depth, out_height, out_width] = grid_shape.try_into().expect("rank checked");
    let output_len = [batch, channels, out_depth, out_height, out_width]
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(InterpolationError::OutputSizeOverflow)?;
    let image = image.to_contiguous();
    let grid = grid.to_contiguous();
    let image_values = image.as_slice();
    let grid_values = grid.as_slice();
    let output_points = out_depth * out_height * out_width;
    let input_points = depth * height * width;
    let mut output = vec![0.0; output_len];

    for b in 0..batch {
        for point in 0..output_points {
            let coordinate = |axis| grid_values[(b * 3 + axis) * output_points + point];
            let neighbours = |value: f32, extent: usize| {
                let lower_value = value.floor();
                let weight = value - lower_value;
                let lower = lower_value.clamp(0.0, (extent - 1) as f32) as usize;
                let upper = (lower_value + 1.0).clamp(0.0, (extent - 1) as f32) as usize;
                (lower, upper, 1.0 - weight, weight)
            };
            let (z0, z1, wz0, wz1) = neighbours(coordinate(0), depth);
            let (y0, y1, wy0, wy1) = neighbours(coordinate(1), height);
            let (x0, x1, wx0, wx1) = neighbours(coordinate(2), width);

            for channel in 0..channels {
                let base = (b * channels + channel) * input_points;
                let at = |z, y, x| image_values[base + (z * height + y) * width + x];
                let xy00 = at(z0, y0, x0) * wx0 + at(z0, y0, x1) * wx1;
                let xy01 = at(z0, y1, x0) * wx0 + at(z0, y1, x1) * wx1;
                let xy10 = at(z1, y0, x0) * wx0 + at(z1, y0, x1) * wx1;
                let xy11 = at(z1, y1, x0) * wx0 + at(z1, y1, x1) * wx1;
                let z0_value = xy00 * wy0 + xy01 * wy1;
                let z1_value = xy10 * wy0 + xy11 * wy1;
                output[(b * channels + channel) * output_points + point] =
                    z0_value * wz0 + z1_value * wz1;
            }
        }
    }

    Ok(Tensor::from_slice_on(
        [batch, channels, out_depth, out_height, out_width],
        &output,
        &B::default(),
    ))
}
