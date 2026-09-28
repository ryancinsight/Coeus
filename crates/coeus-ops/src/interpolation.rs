//! Dimension-generic coordinate-grid interpolation operations.

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_tensor::Tensor;

mod private {
    pub trait Sealed {}
}

/// Static boundary behavior for linear interpolation.
pub trait BoundaryPolicy: private::Sealed + Copy + Default {
    /// Returns lower/upper indices and their interpolation weights.
    ///
    /// `coordinate` must be finite and `extent` must be non-zero.
    fn neighbours<T: Float>(coordinate: T, extent: usize) -> (usize, usize, T, T);
}

/// Replicate the nearest border value outside the image extent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Replicate;

impl private::Sealed for Replicate {}

impl BoundaryPolicy for Replicate {
    fn neighbours<T: Float>(coordinate: T, extent: usize) -> (usize, usize, T, T) {
        let lower_value = coordinate.floor();
        let upper_weight = coordinate - lower_value;
        let zero = T::zero();
        if coordinate < zero {
            return (0, 0, T::one() - upper_weight, upper_weight);
        }

        let last = extent - 1;
        // The coordinate selects a discrete voxel here; converting its floored
        // value to an index is the boundary between scalar arithmetic and the
        // integer address space. The float-to-integer cast saturates, then the
        // original integer extent enforces the valid index range.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Voxel selection converts a finite floored scalar to a discrete index; the result is clamped to the validated integer extent."
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "Negative coordinates are handled above before conversion to an unsigned index."
        )]
        let lower = (Scalar::to_f64(lower_value) as usize).min(last);
        let upper = lower.saturating_add(1).min(last);
        (lower, upper, T::one() - upper_weight, upper_weight)
    }
}

/// Const-dimension marker restricting interpolation monomorphizations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dimension<const D: usize>;

/// Compile-time evidence that an interpolation dimension is implemented.
pub trait SupportedDimension: private::Sealed {}

impl private::Sealed for Dimension<2> {}
impl private::Sealed for Dimension<3> {}
impl SupportedDimension for Dimension<2> {}
impl SupportedDimension for Dimension<3> {}

/// Gradients produced by dimension-generic linear interpolation reverse mode.
///
/// `T` is the scalar type used by the image and coordinate gradients. It
/// defaults to `f32` for existing type annotations.
pub struct InterpolationGradients<B: Backend, T: Scalar = f32> {
    /// Gradient with respect to image values.
    pub image: Tensor<T, B>,
    /// Gradient with respect to sampling coordinates.
    pub grid: Tensor<T, B>,
}

/// Contract failures for [`linear_interpolation`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InterpolationError {
    /// The image rank differs from `[batch, channel, spatial...]`.
    #[error("{dimension}-D interpolation requires image rank {expected}, got {actual}")]
    ImageRank {
        /// Spatial dimension.
        dimension: usize,
        /// Required rank.
        expected: usize,
        /// Supplied rank.
        actual: usize,
    },
    /// The grid rank differs from `[batch, coordinate, spatial...]`.
    #[error("{dimension}-D interpolation requires grid rank {expected}, got {actual}")]
    GridRank {
        /// Spatial dimension.
        dimension: usize,
        /// Required rank.
        expected: usize,
        /// Supplied rank.
        actual: usize,
    },
    /// The grid batch does not match the image batch.
    #[error("linear interpolation batch mismatch: image {image}, grid {grid}")]
    BatchMismatch {
        /// Image batch extent.
        image: usize,
        /// Grid batch extent.
        grid: usize,
    },
    /// The grid coordinate-channel count differs from the spatial dimension.
    #[error("linear interpolation grid requires {expected} coordinate channels, got {actual}")]
    GridChannels {
        /// Spatial dimension.
        expected: usize,
        /// Supplied channel count.
        actual: usize,
    },
    /// An input spatial axis is empty.
    #[error("linear interpolation image spatial axis {axis} is empty")]
    EmptySpatialAxis {
        /// Zero-based spatial axis.
        axis: usize,
    },
    /// A sampling coordinate is NaN or infinite.
    #[error("linear interpolation coordinate axis {axis} at output point {point} is not finite")]
    NonFiniteCoordinate {
        /// Coordinate axis.
        axis: usize,
        /// Flattened output point.
        point: usize,
    },
    /// A shape product overflowed `usize`.
    #[error("linear interpolation element count overflow")]
    SizeOverflow,
    /// The upstream gradient shape does not match the interpolation output.
    #[error("linear interpolation gradient shape mismatch: expected {expected:?}, got {actual:?}")]
    GradientShape {
        /// Shape implied by the image and sampling grid.
        expected: Vec<usize>,
        /// Supplied upstream-gradient shape.
        actual: Vec<usize>,
    },
}

struct Contract {
    batch: usize,
    channels: usize,
    input_spatial: Vec<usize>,
    output_spatial: Vec<usize>,
    input_points: usize,
    output_points: usize,
    output_shape: Vec<usize>,
}

fn checked_product(extents: &[usize]) -> Result<usize, InterpolationError> {
    extents
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(InterpolationError::SizeOverflow)
}

fn validate<const D: usize>(image: &[usize], grid: &[usize]) -> Result<Contract, InterpolationError>
where
    Dimension<D>: SupportedDimension,
{
    let rank = D + 2;
    if image.len() != rank {
        return Err(InterpolationError::ImageRank {
            dimension: D,
            expected: rank,
            actual: image.len(),
        });
    }
    if grid.len() != rank {
        return Err(InterpolationError::GridRank {
            dimension: D,
            expected: rank,
            actual: grid.len(),
        });
    }
    if image[0] != grid[0] {
        return Err(InterpolationError::BatchMismatch {
            image: image[0],
            grid: grid[0],
        });
    }
    if grid[1] != D {
        return Err(InterpolationError::GridChannels {
            expected: D,
            actual: grid[1],
        });
    }
    for (axis, &extent) in image[2..].iter().enumerate() {
        if extent == 0 {
            return Err(InterpolationError::EmptySpatialAxis { axis });
        }
    }
    let input_spatial = image[2..].to_vec();
    let output_spatial = grid[2..].to_vec();
    let input_points = checked_product(&input_spatial)?;
    let output_points = checked_product(&output_spatial)?;
    let output_shape = [image[0], image[1]]
        .into_iter()
        .chain(output_spatial.iter().copied())
        .collect::<Vec<_>>();
    checked_product(&output_shape)?;
    Ok(Contract {
        batch: image[0],
        channels: image[1],
        input_spatial,
        output_spatial,
        input_points,
        output_points,
        output_shape,
    })
}

fn spatial_offset(indices: &[usize], extents: &[usize]) -> usize {
    indices
        .iter()
        .zip(extents)
        .fold(0, |offset, (&index, &extent)| offset * extent + index)
}

/// Sample a 2-D or 3-D image at a voxel-coordinate grid.
///
/// `image` has shape `[batch, channel, spatial...]`; `grid` has shape
/// `[batch, D, output_spatial...]`. Coordinate channels follow image spatial
/// axis order. `D` is restricted to 2 or 3, while `P` selects boundary
/// behavior at compile time with zero runtime storage.
///
/// Image values, coordinates, interpolation weights, and accumulation all use
/// the same `T: Float`; the operation does not change scalar precision.
///
/// # Errors
///
/// Returns [`InterpolationError`] when ranks, batches, coordinate channels,
/// spatial extents, finite-coordinate requirement, or shape products violate
/// the contract.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), coeus_ops::InterpolationError> {
/// use coeus_core::SequentialBackend;
/// use coeus_ops::{linear_interpolation, linear_interpolation_backward, Replicate};
/// use coeus_tensor::Tensor;
///
/// let backend = SequentialBackend;
/// let image = Tensor::from_slice_on(
///     [1, 1, 2, 2],
///     &[0.0_f64, 1.0, 2.0, 3.0],
///     &backend,
/// );
/// let grid = Tensor::from_slice_on([1, 2, 1, 1], &[0.25_f64, 0.75], &backend);
/// let sampled = linear_interpolation::<2, _, _, _>(&image, &grid, Replicate)?;
/// assert_eq!(sampled.as_slice(), &[1.25]);
///
/// let grad_output = Tensor::from_slice_on([1, 1, 1, 1], &[1.0_f64], &backend);
/// let gradients =
///     linear_interpolation_backward::<2, _, _, _>(&image, &grid, &grad_output, Replicate)?;
/// assert_eq!(gradients.grid.as_slice(), &[2.0, 1.0]);
/// # Ok(())
/// # }
/// ```
pub fn linear_interpolation<const D: usize, B, P, T>(
    image: &Tensor<T, B>,
    grid: &Tensor<T, B>,
    _policy: P,
) -> Result<Tensor<T, B>, InterpolationError>
where
    T: Float,
    B: Backend + Default,
    P: BoundaryPolicy,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let contract = validate::<D>(image.shape(), grid.shape())?;
    let image = image.to_contiguous();
    let grid = grid.to_contiguous();
    let image_values = image.as_slice();
    let grid_values = grid.as_slice();
    let output_len = checked_product(&contract.output_shape)?;
    let mut output = vec![T::zero(); output_len];
    let corner_count = 1usize << D;

    for batch in 0..contract.batch {
        for point in 0..contract.output_points {
            let mut neighbours = [(0, 0, T::zero(), T::zero()); D];
            for (axis, entry) in neighbours.iter_mut().enumerate() {
                let coordinate = grid_values[(batch * D + axis) * contract.output_points + point];
                if !coeus_core::Float::is_finite(coordinate) {
                    return Err(InterpolationError::NonFiniteCoordinate { axis, point });
                }
                *entry = P::neighbours(coordinate, contract.input_spatial[axis]);
            }
            for channel in 0..contract.channels {
                let base = (batch * contract.channels + channel) * contract.input_points;
                let mut value = T::zero();
                for corner in 0..corner_count {
                    let mut indices = [0; D];
                    let mut weight = T::one();
                    for (axis, &(lower, upper, lower_weight, upper_weight)) in
                        neighbours.iter().enumerate()
                    {
                        let upper_side = corner & (1 << axis) != 0;
                        indices[axis] = if upper_side { upper } else { lower };
                        weight *= if upper_side {
                            upper_weight
                        } else {
                            lower_weight
                        };
                    }
                    value += image_values[base + spatial_offset(&indices, &contract.input_spatial)]
                        * weight;
                }
                output[(batch * contract.channels + channel) * contract.output_points + point] =
                    value;
            }
        }
    }

    Ok(Tensor::from_slice_on(
        contract.output_shape,
        &output,
        &B::default(),
    ))
}

/// Backpropagate through [`linear_interpolation`].
///
/// Returns gradients for `(image, grid)`. The image derivative scatters to
/// `2^D` neighbours. Each coordinate derivative differentiates the same
/// multilinear polynomial; replicated axes have zero derivative when both
/// neighbours resolve to one border element.
///
/// Gradients, coordinates, weights, and accumulation all use the input scalar
/// type `T`.
///
/// # Errors
///
/// Returns [`InterpolationError`] under the forward contract failures or when
/// `grad_output` does not have the implied output shape. Non-finite sampling
/// coordinates are rejected before derivative arithmetic.
pub fn linear_interpolation_backward<const D: usize, B, P, T>(
    image: &Tensor<T, B>,
    grid: &Tensor<T, B>,
    grad_output: &Tensor<T, B>,
    _policy: P,
) -> Result<InterpolationGradients<B, T>, InterpolationError>
where
    T: Float,
    B: Backend + Default,
    P: BoundaryPolicy,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let contract = validate::<D>(image.shape(), grid.shape())?;
    if grad_output.shape() != contract.output_shape {
        return Err(InterpolationError::GradientShape {
            expected: contract.output_shape,
            actual: grad_output.shape().to_vec(),
        });
    }
    let image = image.to_contiguous();
    let grid = grid.to_contiguous();
    let grad_output = grad_output.to_contiguous();
    let image_values = image.as_slice();
    let grid_values = grid.as_slice();
    let upstream = grad_output.as_slice();
    let mut image_gradient = vec![T::zero(); image_values.len()];
    let mut grid_gradient = vec![T::zero(); grid_values.len()];
    let corner_count = 1usize << D;

    for batch in 0..contract.batch {
        for point in 0..contract.output_points {
            let mut neighbours = [(0, 0, T::zero(), T::zero()); D];
            for (axis, entry) in neighbours.iter_mut().enumerate() {
                let coordinate = grid_values[(batch * D + axis) * contract.output_points + point];
                if !coeus_core::Float::is_finite(coordinate) {
                    return Err(InterpolationError::NonFiniteCoordinate { axis, point });
                }
                *entry = P::neighbours(coordinate, contract.input_spatial[axis]);
            }
            for channel in 0..contract.channels {
                let base = (batch * contract.channels + channel) * contract.input_points;
                let output_index =
                    (batch * contract.channels + channel) * contract.output_points + point;
                let grad = upstream[output_index];
                for corner in 0..corner_count {
                    let mut indices = [0; D];
                    let mut weights = [T::zero(); D];
                    for (axis, &(lower, upper, lower_weight, upper_weight)) in
                        neighbours.iter().enumerate()
                    {
                        let upper_side = corner & (1 << axis) != 0;
                        indices[axis] = if upper_side { upper } else { lower };
                        weights[axis] = if upper_side {
                            upper_weight
                        } else {
                            lower_weight
                        };
                    }
                    let input_index = base + spatial_offset(&indices, &contract.input_spatial);
                    let value = image_values[input_index];
                    let weight = weights.iter().fold(T::one(), |acc, &w| acc * w);
                    image_gradient[input_index] += grad * weight;
                    for axis in 0..D {
                        let (lower, upper, _, _) = neighbours[axis];
                        if lower == upper {
                            continue;
                        }
                        let sign = if corner & (1 << axis) != 0 {
                            T::one()
                        } else {
                            T::zero() - T::one()
                        };
                        let other_weight = weights
                            .iter()
                            .enumerate()
                            .filter(|(candidate, _)| *candidate != axis)
                            .fold(T::one(), |acc, (_, &w)| acc * w);
                        let index = (batch * D + axis) * contract.output_points + point;
                        grid_gradient[index] += grad * value * sign * other_weight;
                    }
                }
            }
        }
    }

    let backend = B::default();
    let image_shape = [contract.batch, contract.channels]
        .into_iter()
        .chain(contract.input_spatial)
        .collect::<Vec<_>>();
    let grid_shape = [contract.batch, D]
        .into_iter()
        .chain(contract.output_spatial)
        .collect::<Vec<_>>();
    Ok(InterpolationGradients {
        image: Tensor::from_slice_on(image_shape, &image_gradient, &backend),
        grid: Tensor::from_slice_on(grid_shape, &grid_gradient, &backend),
    })
}
