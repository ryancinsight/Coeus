//! Finite-difference checks for the spatial resize ops
//! (`coeus_autograd::resize_1d` / `resize_2d`) in every mode.
//!
//! All four modes are linear in the input *values* — nearest is piecewise
//! constant, bilinear/bicubic are fixed-weight combinations, and area is a box
//! mean — and the tap set is chosen from the shapes alone, never from the
//! values. So a central difference probes the same linear map the backward
//! scatters through: there is no kink to straddle, and the comparison is exact
//! up to `ε^(2/3)`.
//!
//! Both resize directions are checked per mode: downsampling makes several
//! output positions land on the same source element (the accumulation path),
//! while upsampling gives interior positions two or four distinct taps each
//! (the blend path) and, for `bicubic`, negative weights that reach the clamped
//! border taps.

use super::{tensor, weighted, weighting, GradcheckScalar};
use coeus_autograd::{gradcheck, resize_1d, resize_2d};
use coeus_ops::ResizeMode;

/// `[1, 1, in_len] → [1, 1, out_len]` in `mode`, differentiated w.r.t. the input.
fn resize_1d_case<T: GradcheckScalar>(mode: ResizeMode, in_len: usize, out_len: usize, phase: f64) {
    let x = tensor::<T>(&[1, 1, in_len], phase);
    let w = weighting::<T>(&[1, 1, out_len]);
    gradcheck(&[x], |v| {
        let y = resize_1d(&v[0], out_len, mode)
            .expect("invariant: valid resize fixture completes forward");
        weighted(&y, &w)
    })
    .expect("resize_1d backward must match central differences");
}

/// `[1, 2, in_h, in_w] → [1, 2, out_h, out_w]` in `mode`.
fn resize_2d_case<T: GradcheckScalar>(
    mode: ResizeMode,
    input: (usize, usize),
    output: (usize, usize),
    phase: f64,
) {
    let (in_h, in_w) = input;
    let (out_h, out_w) = output;
    let x = tensor::<T>(&[1, 2, in_h, in_w], phase);
    let w = weighting::<T>(&[1, 2, out_h, out_w]);
    gradcheck(&[x], |v| {
        let y = resize_2d(&v[0], out_h, out_w, mode)
            .expect("invariant: valid resize fixture completes forward");
        weighted(&y, &w)
    })
    .expect("resize_2d backward must match central differences");
}

#[test]
fn resize_1d_nearest_backward_matches_finite_differences() {
    // 6 → 3 drops half the source; 4 → 7 copies several sources twice.
    resize_1d_case::<f64>(ResizeMode::Nearest, 6, 3, 0.17);
    resize_1d_case::<f64>(ResizeMode::Nearest, 4, 7, 0.29);
    resize_1d_case::<f32>(ResizeMode::Nearest, 6, 3, 0.17);
    resize_1d_case::<f32>(ResizeMode::Nearest, 4, 7, 0.29);
}

#[test]
fn resize_1d_bilinear_backward_matches_finite_differences() {
    resize_1d_case::<f64>(ResizeMode::Bilinear, 6, 3, 0.19);
    resize_1d_case::<f64>(ResizeMode::Bilinear, 4, 7, 0.31);
    resize_1d_case::<f32>(ResizeMode::Bilinear, 6, 3, 0.19);
    resize_1d_case::<f32>(ResizeMode::Bilinear, 4, 7, 0.31);
}

#[test]
fn resize_1d_bicubic_backward_matches_finite_differences() {
    // The 4 → 7 case puts the first output at a negative source coordinate, so
    // the clamped border taps carry the cubic kernel's negative weights.
    resize_1d_case::<f64>(ResizeMode::Bicubic, 6, 3, 0.23);
    resize_1d_case::<f64>(ResizeMode::Bicubic, 4, 7, 0.37);
    resize_1d_case::<f32>(ResizeMode::Bicubic, 6, 3, 0.23);
    resize_1d_case::<f32>(ResizeMode::Bicubic, 4, 7, 0.37);
}

#[test]
fn resize_1d_area_backward_matches_finite_differences() {
    // 6 → 4 gives overlapping, unequal-length boxes (the variable-tap path).
    resize_1d_case::<f64>(ResizeMode::Area, 6, 3, 0.41);
    resize_1d_case::<f64>(ResizeMode::Area, 6, 4, 0.43);
    resize_1d_case::<f32>(ResizeMode::Area, 6, 3, 0.41);
    resize_1d_case::<f32>(ResizeMode::Area, 6, 4, 0.43);
}

#[test]
fn resize_2d_nearest_backward_matches_finite_differences() {
    resize_2d_case::<f64>(ResizeMode::Nearest, (4, 4), (3, 3), 0.47);
    resize_2d_case::<f64>(ResizeMode::Nearest, (3, 3), (5, 4), 0.53);
    resize_2d_case::<f32>(ResizeMode::Nearest, (4, 4), (3, 3), 0.47);
    resize_2d_case::<f32>(ResizeMode::Nearest, (3, 3), (5, 4), 0.53);
}

#[test]
fn resize_2d_bilinear_backward_matches_finite_differences() {
    resize_2d_case::<f64>(ResizeMode::Bilinear, (4, 4), (3, 3), 0.59);
    resize_2d_case::<f64>(ResizeMode::Bilinear, (3, 3), (5, 4), 0.61);
    resize_2d_case::<f32>(ResizeMode::Bilinear, (4, 4), (3, 3), 0.59);
    resize_2d_case::<f32>(ResizeMode::Bilinear, (3, 3), (5, 4), 0.61);
}

#[test]
fn resize_2d_bicubic_backward_matches_finite_differences() {
    resize_2d_case::<f64>(ResizeMode::Bicubic, (4, 4), (3, 3), 0.67);
    resize_2d_case::<f64>(ResizeMode::Bicubic, (3, 3), (5, 4), 0.71);
    resize_2d_case::<f32>(ResizeMode::Bicubic, (4, 4), (3, 3), 0.67);
    resize_2d_case::<f32>(ResizeMode::Bicubic, (3, 3), (5, 4), 0.71);
}

#[test]
fn resize_2d_area_backward_matches_finite_differences() {
    resize_2d_case::<f64>(ResizeMode::Area, (4, 4), (3, 3), 0.73);
    resize_2d_case::<f64>(ResizeMode::Area, (6, 6), (4, 3), 0.79);
    resize_2d_case::<f32>(ResizeMode::Area, (4, 4), (3, 3), 0.73);
    resize_2d_case::<f32>(ResizeMode::Area, (6, 6), (4, 3), 0.79);
}
