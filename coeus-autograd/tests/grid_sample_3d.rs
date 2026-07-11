//! Verification for the differentiable 3-D grid-sample (trilinear warp).
//!
//! Two independent oracles:
//!   * Value: coordinates that resolve to an exact voxel center return that
//!     voxel; a per-axis midpoint returns the mean of the 8 corner voxels.
//!     These follow from the trilinear weights alone, independent of the
//!     implementation's summation order.
//!   * Gradient: central finite differences w.r.t. BOTH `input` and `grid`.
//!     Within a voxel cell the trilinear map is linear in each perturbed scalar
//!     individually, so the central-difference truncation error is exactly zero
//!     and the only residual is `f32` round-off `~ eps * |loss| / h`. With the
//!     scaled `O(1)` image below (`|loss| < ~25`), `eps_f32 = 1.19e-7`, and
//!     `h = 1/64`, that floor is `~ 1.19e-7 * 25 / (1/64) ~= 1.9e-4`; the `1e-3`
//!     tolerance is that floor with a ~5x safety margin for accumulation across
//!     the reduction. Base coordinates sit >= 0.2 (pixel units) from any integer
//!     so `+-h` (pixel shift 1/64 ~= 0.0156) never crosses a cell boundary.

use coeus_autograd::{grid_sample_3d, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

const D: usize = 3;
const H: usize = 3;
const W: usize = 3;
const C: usize = 2;

/// Deterministic `O(1)` image pattern with distinct per-voxel values.
fn image_value(c: usize, z: usize, y: usize, x: usize) -> f32 {
    0.1 * (c * (D * H * W) + z * (H * W) + y * W + x) as f32
}

fn image_data() -> Vec<f32> {
    let mut data = Vec::with_capacity(C * D * H * W);
    for c in 0..C {
        for z in 0..D {
            for y in 0..H {
                for x in 0..W {
                    data.push(image_value(c, z, y, x));
                }
            }
        }
    }
    data
}

#[test]
fn samples_exact_voxel_center() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on([1, C, D, H, W], &image_data(), &backend),
        false,
    );
    // align_corners: pixel = (coord+1)/2*(extent-1); coord 0 -> pixel 1 (center).
    let grid = Var::new(
        Tensor::from_slice_on([1, 1, 1, 1, 3], &[0.0, 0.0, 0.0], &backend),
        false,
    );
    let out = grid_sample_3d(&image, &grid);
    // Output (N, C, 1, 1, 1): each channel returns image[c, 1, 1, 1] exactly.
    let got = out.tensor.as_slice();
    for (c, &value) in got.iter().take(C).enumerate() {
        let expected = image_value(c, 1, 1, 1);
        assert!(
            (value - expected).abs() <= 1e-6,
            "channel {c}: got {value}, expected {expected}"
        );
    }
}

#[test]
fn samples_corner_block_midpoint_as_mean() {
    let backend = MoiraiBackend;
    let image = Var::new(
        Tensor::from_slice_on([1, C, D, H, W], &image_data(), &backend),
        false,
    );
    // coord -0.5 -> pixel 0.5 in each axis: midpoint of the {0,1}^3 voxel block.
    let grid = Var::new(
        Tensor::from_slice_on([1, 1, 1, 1, 3], &[-0.5, -0.5, -0.5], &backend),
        false,
    );
    let out = grid_sample_3d(&image, &grid);
    let got = out.tensor.as_slice();
    for (c, &value) in got.iter().take(C).enumerate() {
        let mut mean = 0.0f32;
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    mean += image_value(c, z, y, x);
                }
            }
        }
        mean /= 8.0;
        assert!(
            (value - mean).abs() <= 1e-5,
            "channel {c}: got {value}, expected mean {mean}"
        );
    }
}

/// Two interior sample points, each >= 0.2 pixel-units from every voxel edge.
const GRID: [f32; 6] = [
    -0.3, 0.3, -0.4, // point 0: pixels (x,y,z) = (0.7, 1.3, 0.6)
    0.4, -0.4, 0.2, // point 1: pixels (x,y,z) = (1.4, 0.6, 1.2)
];
const GRID_SHAPE: [usize; 5] = [1, 1, 1, 2, 3];

/// Scalar loss `sum(grid_sample_3d(image, grid))` with grad tracking off.
fn loss(image: &[f32], grid: &[f32]) -> f64 {
    let backend = MoiraiBackend;
    let input = Var::new(
        Tensor::from_slice_on([1, C, D, H, W], image, &backend),
        false,
    );
    let g = Var::new(Tensor::from_slice_on(GRID_SHAPE, grid, &backend), false);
    let out = grid_sample_3d(&input, &g);
    out.tensor.as_slice().iter().map(|&v| f64::from(v)).sum()
}

#[test]
fn input_gradient_matches_central_difference() {
    let backend = MoiraiBackend;
    let image_vec = image_data();
    let input = Var::new(
        Tensor::from_slice_on([1, C, D, H, W], &image_vec, &backend),
        true,
    );
    let grid = Var::new(Tensor::from_slice_on(GRID_SHAPE, &GRID, &backend), true);
    let out = grid_sample_3d(&input, &grid);
    sum(&out).backward();
    let analytic = input.grad().expect("tracked input gradient");
    let analytic = analytic.as_slice();

    let h = 1.0f64 / 64.0;
    for i in 0..image_vec.len() {
        let mut plus = image_vec.clone();
        let mut minus = image_vec.clone();
        plus[i] += h as f32;
        minus[i] -= h as f32;
        let fd = (loss(&plus, &GRID) - loss(&minus, &GRID)) / (2.0 * h);
        assert!(
            (f64::from(analytic[i]) - fd).abs() <= 1e-3,
            "input[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}

#[test]
fn grid_gradient_matches_central_difference() {
    let backend = MoiraiBackend;
    let image_vec = image_data();
    let input = Var::new(
        Tensor::from_slice_on([1, C, D, H, W], &image_vec, &backend),
        true,
    );
    let grid = Var::new(Tensor::from_slice_on(GRID_SHAPE, &GRID, &backend), true);
    let out = grid_sample_3d(&input, &grid);
    sum(&out).backward();
    let analytic = grid.grad().expect("tracked grid gradient");
    let analytic = analytic.as_slice();

    let h = 1.0f64 / 64.0;
    for i in 0..GRID.len() {
        let mut plus = GRID;
        let mut minus = GRID;
        plus[i] += h as f32;
        minus[i] -= h as f32;
        let fd = (loss(&image_vec, &plus) - loss(&image_vec, &minus)) / (2.0 * h);
        assert!(
            (f64::from(analytic[i]) - fd).abs() <= 1e-3,
            "grid[{i}]: analytic {}, finite-diff {fd}",
            analytic[i]
        );
    }
}
