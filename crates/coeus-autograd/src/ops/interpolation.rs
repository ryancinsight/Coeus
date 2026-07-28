//! Differentiable dimension-generic coordinate-grid interpolation.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::{
    linear_interpolation_backward, BoundaryPolicy, Dimension, InterpolationError, Replicate,
    SupportedDimension,
};
use coeus_tensor::Tensor;
use std::{marker::PhantomData, sync::Arc};

/// Reverse-mode node for linear sampling.
pub struct LinearInterpolationNode<const D: usize, B, P = Replicate>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy,
{
    /// Accumulated output gradient.
    pub output_grad: Arc<GradBuffer<f32, B>>,
    /// Image and sampling-grid variables.
    pub inputs: Vec<Var<f32, B>>,
    /// Saved image values required by the coordinate derivative.
    pub image: Tensor<f32, B>,
    /// Saved sampling coordinates required by both derivatives.
    pub grid: Tensor<f32, B>,
    policy: PhantomData<P>,
}

impl<const D: usize, B, P> BackwardNode<f32, B> for LinearInterpolationNode<D, B, P>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn op_name(&self) -> &'static str {
        "linear_interpolation"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<f32, B>,
        input_grads: &[Option<Arc<GradBuffer<f32, B>>>],
    ) -> Result<(), B::Error> {
        let updates = linear_interpolation_backward::<D, _, P>(
            &self.image,
            &self.grid,
            grad_out,
            P::default(),
        )
        .map_err(|error| {
            B::Error::from(coeus_core::BackendError::Storage {
                operation: "linear_interpolation_backward",
                reason: error.to_string(),
            })
        })?;
        let backend = B::default();
        if let Some(Some(gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &updates.image, &backend)?;
        }
        if let Some(Some(gradient)) = input_grads.get(1) {
            coeus_ops::add_assign(gradient.write(), &updates.grid, &backend)?;
        }
        Ok(())
    }
}

/// Sample a 2-D or 3-D image while tracking image and coordinate gradients.
///
/// `D` selects the spatial dimension and `P` is a zero-sized boundary policy.
///
/// # Errors
///
/// Returns [`InterpolationError`] when image or grid shape violates the
/// dimension-generic interpolation contract.
pub fn linear_interpolation<const D: usize, B, P>(
    image: &Var<f32, B>,
    grid: &Var<f32, B>,
    policy: P,
) -> Result<Var<f32, B>, InterpolationError>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    P: BoundaryPolicy + Send + Sync + 'static,
    Dimension<D>: SupportedDimension,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let output = coeus_ops::linear_interpolation::<D, _, P>(&image.tensor, &grid.tensor, policy)?;
    let requires_grad =
        crate::grad_mode::should_track_var(image) || crate::grad_mode::should_track_var(grid);
    if !requires_grad {
        return Ok(Var::new(output, false));
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(output.shape(), &backend)));
    let node = LinearInterpolationNode::<D, B, P> {
        output_grad: output_grad.clone(),
        inputs: vec![image.clone(), grid.clone()],
        image: image.tensor.clone(),
        grid: grid.tensor.clone(),
        policy: PhantomData,
    };
    Ok(Var {
        tensor: output,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    })
}

// ── 3-D grid-sample (trilinear warp), PyTorch `grid_sample` semantics ──
//
// Convention (pinned; do not mix with `linear_interpolation`'s voxel-coordinate
// convention):
//   * input  shape: (N, C, D, H, W)          — image / feature volume.
//   * grid   shape: (N, D_out, H_out, W_out, 3), last dim = (x, y, z), each a
//     NORMALIZED coordinate in [-1, 1].
//   * output shape: (N, C, D_out, H_out, W_out).
//   * Coordinate-to-axis mapping (PyTorch order): x → W (innermost spatial),
//     y → H, z → D (outermost spatial). The last grid axis is (x, y, z), the
//     REVERSE of the spatial-axis order (D, H, W).
//   * Unnormalization is `align_corners = true`:
//       pixel = (coord + 1) / 2 * (extent - 1)
//     so coord = -1 maps to index 0 and coord = +1 maps to index `extent - 1`.
//   * Padding mode: zeros. A trilinear corner whose voxel index is outside
//     `[0, extent)` contributes value 0 (and 0 to the coordinate derivative).
//   * A non-finite sampling coordinate produces 0 output and 0 gradient at that
//     output point (a defined, panic-free behavior; no NaN index arithmetic).
//
// Interpolation is trilinear over the 8 neighboring voxels. Output is linear in
// each `input` value (fixed weights) and, within a voxel cell, linear in each
// normalized coordinate individually, which the finite-difference gradient
// tests exploit for an exact within-cell oracle.

/// Spatial extents `(D, H, W)` of the input volume.
#[derive(Clone, Copy)]
struct VolumeShape {
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
}

/// Spatial extents `(D_out, H_out, W_out)` of the sampling grid.
#[derive(Clone, Copy)]
struct GridShape {
    out_depth: usize,
    out_height: usize,
    out_width: usize,
}

/// Per-axis neighbor pair: `(index0, index1, weight0, weight1)`.
#[derive(Clone, Copy)]
struct AxisNeighbours {
    lower: isize,
    upper: isize,
    lower_weight: f32,
    upper_weight: f32,
}

impl AxisNeighbours {
    /// Un-normalize a `[-1, 1]` coordinate to a pixel position and split it into
    /// its two trilinear neighbors under `align_corners = true`.
    #[inline]
    fn from_normalized(coordinate: f32, extent: usize) -> Self {
        let pixel = (coordinate + 1.0) * 0.5 * (extent as f32 - 1.0);
        let lower_pixel = pixel.floor();
        let upper_weight = pixel - lower_pixel;
        Self {
            lower: lower_pixel as isize,
            upper: lower_pixel as isize + 1,
            lower_weight: 1.0 - upper_weight,
            upper_weight,
        }
    }
}

/// d(pixel)/d(normalized coordinate) for `align_corners = true`.
#[inline]
fn coordinate_scale(extent: usize) -> f32 {
    0.5 * (extent as f32 - 1.0)
}

fn parse_shapes(input: &[usize], grid: &[usize]) -> (VolumeShape, GridShape) {
    assert_eq!(
        input.len(),
        5,
        "grid_sample_3d: input must be rank-5 (N, C, D, H, W), got {input:?}"
    );
    assert_eq!(
        grid.len(),
        5,
        "grid_sample_3d: grid must be rank-5 (N, D_out, H_out, W_out, 3), got {grid:?}"
    );
    assert_eq!(
        grid[4], 3,
        "grid_sample_3d: grid last dim must be 3 (x, y, z), got {}",
        grid[4]
    );
    assert_eq!(
        input[0], grid[0],
        "grid_sample_3d: batch mismatch: input {} vs grid {}",
        input[0], grid[0]
    );
    assert!(
        input[2] > 0 && input[3] > 0 && input[4] > 0,
        "grid_sample_3d: input spatial extents must be non-empty, got {input:?}"
    );
    (
        VolumeShape {
            batch: input[0],
            channels: input[1],
            depth: input[2],
            height: input[3],
            width: input[4],
        },
        GridShape {
            out_depth: grid[1],
            out_height: grid[2],
            out_width: grid[3],
        },
    )
}

/// Contiguous element offset into an `(N, C, D, H, W)` volume.
#[inline]
fn volume_offset(vol: &VolumeShape, n: usize, c: usize, iz: usize, iy: usize, ix: usize) -> usize {
    (((n * vol.channels + c) * vol.depth + iz) * vol.height + iy) * vol.width + ix
}

/// Fetch `input[n, c, iz, iy, ix]` under zeros padding (0 when out of bounds).
#[inline]
fn sample_zeros(
    input: &[f32],
    vol: &VolumeShape,
    n: usize,
    c: usize,
    iz: isize,
    iy: isize,
    ix: isize,
) -> f32 {
    if iz < 0
        || iy < 0
        || ix < 0
        || iz >= vol.depth as isize
        || iy >= vol.height as isize
        || ix >= vol.width as isize
    {
        return 0.0;
    }
    input[volume_offset(vol, n, c, iz as usize, iy as usize, ix as usize)]
}

fn grid_sample_3d_forward<B>(input: &Tensor<f32, B>, grid: &Tensor<f32, B>) -> Tensor<f32, B>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (vol, out) = parse_shapes(input.shape(), grid.shape());
    let input = input.to_contiguous();
    let grid = grid.to_contiguous();
    let image = input.as_slice();
    let coords = grid.as_slice();

    let out_points = out.out_depth * out.out_height * out.out_width;
    let output_shape = [
        vol.batch,
        vol.channels,
        out.out_depth,
        out.out_height,
        out.out_width,
    ];
    let mut output = vec![0.0f32; vol.batch * vol.channels * out_points];

    for n in 0..vol.batch {
        for point in 0..out_points {
            let coord_base = (n * out_points + point) * 3;
            let gx = coords[coord_base];
            let gy = coords[coord_base + 1];
            let gz = coords[coord_base + 2];
            if !(gx.is_finite() && gy.is_finite() && gz.is_finite()) {
                continue; // zeros for a non-finite coordinate
            }
            let nx = AxisNeighbours::from_normalized(gx, vol.width);
            let ny = AxisNeighbours::from_normalized(gy, vol.height);
            let nz = AxisNeighbours::from_normalized(gz, vol.depth);
            let zi = [nz.lower, nz.upper];
            let zw = [nz.lower_weight, nz.upper_weight];
            let yi = [ny.lower, ny.upper];
            let yw = [ny.lower_weight, ny.upper_weight];
            let xi = [nx.lower, nx.upper];
            let xw = [nx.lower_weight, nx.upper_weight];
            for c in 0..vol.channels {
                let mut value = 0.0f32;
                for a in 0..2 {
                    for b in 0..2 {
                        for e in 0..2 {
                            let weight = zw[a] * yw[b] * xw[e];
                            if weight == 0.0 {
                                continue;
                            }
                            value += weight * sample_zeros(image, &vol, n, c, zi[a], yi[b], xi[e]);
                        }
                    }
                }
                output[(n * vol.channels + c) * out_points + point] = value;
            }
        }
    }

    Tensor::from_slice_on(output_shape, &output, &B::default())
}

/// Reverse mode for [`grid_sample_3d`]: gradients for `(input, grid)`.
fn grid_sample_3d_backward<B>(
    input: &Tensor<f32, B>,
    grid: &Tensor<f32, B>,
    grad_output: &Tensor<f32, B>,
) -> (Tensor<f32, B>, Tensor<f32, B>)
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (vol, out) = parse_shapes(input.shape(), grid.shape());
    let input = input.to_contiguous();
    let grid = grid.to_contiguous();
    let grad_output = grad_output.to_contiguous();
    let image = input.as_slice();
    let coords = grid.as_slice();
    let upstream = grad_output.as_slice();

    let out_points = out.out_depth * out.out_height * out.out_width;
    let mut input_grad = vec![0.0f32; image.len()];
    let mut grid_grad = vec![0.0f32; coords.len()];

    let scale_x = coordinate_scale(vol.width);
    let scale_y = coordinate_scale(vol.height);
    let scale_z = coordinate_scale(vol.depth);

    for n in 0..vol.batch {
        for point in 0..out_points {
            let coord_base = (n * out_points + point) * 3;
            let gx = coords[coord_base];
            let gy = coords[coord_base + 1];
            let gz = coords[coord_base + 2];
            if !(gx.is_finite() && gy.is_finite() && gz.is_finite()) {
                continue;
            }
            let nx = AxisNeighbours::from_normalized(gx, vol.width);
            let ny = AxisNeighbours::from_normalized(gy, vol.height);
            let nz = AxisNeighbours::from_normalized(gz, vol.depth);
            let zi = [nz.lower, nz.upper];
            let zw = [nz.lower_weight, nz.upper_weight];
            let yi = [ny.lower, ny.upper];
            let yw = [ny.lower_weight, ny.upper_weight];
            let xi = [nx.lower, nx.upper];
            let xw = [nx.lower_weight, nx.upper_weight];

            for c in 0..vol.channels {
                let g = upstream[(n * vol.channels + c) * out_points + point];
                if g == 0.0 {
                    continue;
                }
                // ∂/∂input: scatter the corner-weighted output gradient.
                for a in 0..2 {
                    for b in 0..2 {
                        for e in 0..2 {
                            let (iz, iy, ix) = (zi[a], yi[b], xi[e]);
                            if iz < 0
                                || iy < 0
                                || ix < 0
                                || iz >= vol.depth as isize
                                || iy >= vol.height as isize
                                || ix >= vol.width as isize
                            {
                                continue; // zeros padding: no gradient to a phantom voxel
                            }
                            let weight = zw[a] * yw[b] * xw[e];
                            input_grad[volume_offset(
                                &vol,
                                n,
                                c,
                                iz as usize,
                                iy as usize,
                                ix as usize,
                            )] += g * weight;
                        }
                    }
                }
                // ∂/∂grid: differentiate the trilinear polynomial w.r.t. each
                // pixel coordinate (∂w_upper/∂pixel = +1, ∂w_lower/∂pixel = -1),
                // then chain through d(pixel)/d(normalized) = scale.
                let mut d_pixel_x = 0.0f32;
                let mut d_pixel_y = 0.0f32;
                let mut d_pixel_z = 0.0f32;
                for a in 0..2 {
                    for b in 0..2 {
                        let upper = sample_zeros(image, &vol, n, c, zi[a], yi[b], nx.upper);
                        let lower = sample_zeros(image, &vol, n, c, zi[a], yi[b], nx.lower);
                        d_pixel_x += zw[a] * yw[b] * (upper - lower);
                    }
                }
                for a in 0..2 {
                    for e in 0..2 {
                        let upper = sample_zeros(image, &vol, n, c, zi[a], ny.upper, xi[e]);
                        let lower = sample_zeros(image, &vol, n, c, zi[a], ny.lower, xi[e]);
                        d_pixel_y += zw[a] * xw[e] * (upper - lower);
                    }
                }
                for b in 0..2 {
                    for e in 0..2 {
                        let upper = sample_zeros(image, &vol, n, c, nz.upper, yi[b], xi[e]);
                        let lower = sample_zeros(image, &vol, n, c, nz.lower, yi[b], xi[e]);
                        d_pixel_z += yw[b] * xw[e] * (upper - lower);
                    }
                }
                grid_grad[coord_base] += g * d_pixel_x * scale_x;
                grid_grad[coord_base + 1] += g * d_pixel_y * scale_y;
                grid_grad[coord_base + 2] += g * d_pixel_z * scale_z;
            }
        }
    }

    let backend = B::default();
    (
        Tensor::from_slice_on(input.shape().to_vec(), &input_grad, &backend),
        Tensor::from_slice_on(grid.shape().to_vec(), &grid_grad, &backend),
    )
}

/// Reverse-mode node for [`grid_sample_3d`].
struct GridSample3dNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
{
    output_grad: Arc<GradBuffer<f32, B>>,
    inputs: Vec<Var<f32, B>>,
    input: Tensor<f32, B>,
    grid: Tensor<f32, B>,
}

impl<B> BackwardNode<f32, B> for GridSample3dNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn op_name(&self) -> &'static str {
        "grid_sample_3d"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<f32, B>,
        input_grads: &[Option<Arc<GradBuffer<f32, B>>>],
    ) -> Result<(), B::Error> {
        let (input_grad, grid_grad) = grid_sample_3d_backward(&self.input, &self.grid, grad_out);
        let backend = B::default();
        if let Some(Some(gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &input_grad, &backend)?;
        }
        if let Some(Some(gradient)) = input_grads.get(1) {
            coeus_ops::add_assign(gradient.write(), &grid_grad, &backend)?;
        }
        Ok(())
    }
}

/// Differentiable 3-D grid-sample (trilinear warp) with PyTorch `grid_sample`
/// semantics.
///
/// Samples `input` at the normalized coordinates in `grid`, trilinearly
/// interpolating the 8 neighboring voxels, and tracks gradients for BOTH
/// `input` and `grid`. The `grid` gradient — the sensitivity of each sample to
/// its sampling coordinate — is what deformable-registration optimization
/// requires.
///
/// # Convention (pinned)
/// - `input`: `(N, C, D, H, W)`.
/// - `grid`: `(N, D_out, H_out, W_out, 3)`, last dim `(x, y, z)` normalized to
///   `[-1, 1]`; `x → W`, `y → H`, `z → D` (PyTorch axis order, reverse of the
///   spatial `(D, H, W)` order).
/// - `output`: `(N, C, D_out, H_out, W_out)`.
/// - `align_corners = true`: `pixel = (coord + 1) / 2 * (extent - 1)`.
/// - Padding mode: zeros — an out-of-bounds trilinear corner contributes 0 to
///   both the value and the coordinate derivative.
/// - A non-finite coordinate yields 0 output and 0 gradient at that point.
///
/// # Precision
/// Concrete `f32`, matching the interpolation subsystem
/// ([`linear_interpolation`]). All arithmetic and accumulation run in `f32`
/// (no widen/narrow), so the op is honestly single-precision rather than a
/// generic body that would cast to a fixed type.
///
/// # Panics
/// If `input` is not rank-5, `grid` is not rank-5 with last dim 3, the batch
/// extents differ, or an `input` spatial extent is zero.
#[must_use]
pub fn grid_sample_3d<B>(input: &Var<f32, B>, grid: &Var<f32, B>) -> Var<f32, B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let output = grid_sample_3d_forward(&input.tensor, &grid.tensor);
    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(grid);
    if !requires_grad {
        return Var::new(output, false);
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(output.shape(), &backend)));
    let node = GridSample3dNode::<B> {
        output_grad: output_grad.clone(),
        inputs: vec![input.clone(), grid.clone()],
        input: input.tensor.clone(),
        grid: grid.tensor.clone(),
    };
    Var {
        tensor: output,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    }
}
