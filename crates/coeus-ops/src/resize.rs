//! Spatial resize (nearest / bilinear / bicubic / area) with reverse mode.
//!
//! Conventions match `torch.nn.functional.interpolate` with
//! `align_corners = False` (the half-pixel convention):
//!
//! - 1-D: `[N, C, L]` → `[N, C, new_L]`
//! - 2-D: `[N, C, H, W]` → `[N, C, new_H, new_W]`
//!
//! All modes are separable, so every output position is a weighted sum of
//! source positions per axis. `TapTable::fill` is the single source of truth
//! for those positions and weights: forward and reverse mode both read it, so a
//! gradient can never drift from the value it differentiates.
//!
//! The modes differ only in how the taps are chosen:
//!
//! - [`ResizeMode::Nearest`] — one tap: `floor(i * scale)`, which is torch's
//!   `nearest`. It does **not** half-pixel-correct: that is `nearest-exact`,
//!   whose `floor((i + 0.5) * scale)` gives different indices for a
//!   non-integer size ratio (`3 → 5` picks `0,1,1,2,2` instead of
//!   `0,0,1,1,2`).
//! - [`ResizeMode::Bilinear`] — two taps around the half-pixel source
//!   coordinate, which is clamped at 0 first, so a border output reuses the edge
//!   element with weight 1.
//! - [`ResizeMode::Bicubic`] — four taps, cubic convolution with
//!   `a = -0.75`. The source coordinate is *not* clamped (torch passes
//!   `cubic = true` to `area_pixel_compute_source_index`), so negative weights
//!   reach the clamped border taps and can overshoot.
//! - [`ResizeMode::Area`] — every source element inside the output's box
//!   `[floor(i·in/out), ceil((i+1)·in/out))`, each with equal weight — the
//!   `adaptive_avg_pool` contract torch uses for `mode='area'`.
//!
//! Reverse mode ([`resize_1d_backward`], [`resize_2d_backward`]) scatters each
//! output gradient back through the same taps and accumulates into the source
//! element, since every mode is linear (or, for nearest, piecewise constant) in
//! the source values.

use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement};
use coeus_tensor::Tensor;

/// Cubic convolution coefficient `a` used by torch's bicubic resize.
const CUBIC_COEFFICIENT: f64 = -0.75;

/// Resize mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Nearest-neighbour — each output position copies one source position.
    Nearest,
    /// Linear (bilinear in 2-D) — each output position blends two source
    /// positions per spatial axis.
    Bilinear,
    /// Cubic (bicubic in 2-D) — four source positions per spatial axis, cubic
    /// convolution with `a = -0.75`. Weights can be negative, so the result may
    /// overshoot the source range at edges.
    Bicubic,
    /// Area — average of every source element inside the output position's box.
    Area,
}

/// Cubic convolution kernel: `|x| <= 1` uses the near lobe, `1 < |x| < 2` the
/// far lobe, and the tap contributes nothing beyond 2.
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        (CUBIC_COEFFICIENT + 2.0) * x.powi(3) - (CUBIC_COEFFICIENT + 3.0) * x.powi(2) + 1.0
    } else if x < 2.0 {
        CUBIC_COEFFICIENT * x.powi(3) - 5.0 * CUBIC_COEFFICIENT * x.powi(2)
            + 8.0 * CUBIC_COEFFICIENT * x
            - 4.0 * CUBIC_COEFFICIENT
    } else {
        0.0
    }
}

/// Reusable per-axis tap table.
///
/// One table per axis is held across the whole resize, so the steady-state loop
/// allocates nothing, while still allowing [`ResizeMode::Area`] to emit a
/// variable number of taps (up to the downsample factor).
#[derive(Default)]
struct TapTable {
    index: Vec<usize>,
    weight: Vec<f64>,
}

impl TapTable {
    /// Replace the table contents with the taps for one output position.
    ///
    /// `out_index` must be less than `out_len`, and `in_len`/`out_len` must be
    /// non-zero. `scale` is the sampling step in source units per output
    /// position: `in_len / out_len` for an explicit `size`, but `1 /
    /// scale_factor` when the caller resized by a factor — torch keeps the
    /// requested factor rather than recomputing it from the rounded output
    /// size, so the two differ for a fractional factor.
    fn fill(
        &mut self,
        out_index: usize,
        in_len: usize,
        out_len: usize,
        mode: ResizeMode,
        scale: f64,
    ) {
        self.index.clear();
        self.weight.clear();
        let half_pixel = (out_index as f64 + 0.5) * scale;
        match mode {
            ResizeMode::Nearest => {
                // `floor(i * scale)` — no `+ 0.5`. torch's `nearest` kernel
                // copies the cell the output cell *starts* in, so the two
                // agree on every integer ratio and diverge as soon as the
                // ratio is fractional.
                let source = ((out_index as f64 * scale) as usize).min(in_len - 1);
                self.index.push(source);
                self.weight.push(1.0);
            }
            ResizeMode::Bilinear => {
                // Clamp the source coordinate before splitting it: torch's
                // `area_pixel_compute_source_index` returns 0 for a negative
                // source, so a border output reuses the edge element with
                // weight 1 instead of blending against it fractionally.
                let source = (half_pixel - 0.5).max(0.0);
                let lower = source.floor();
                let lower_index = lower as usize;
                let upper_weight = source - lower;
                self.index.push(lower_index);
                self.weight.push(1.0 - upper_weight);
                self.index.push((lower_index + 1).min(in_len - 1));
                self.weight.push(upper_weight);
            }
            ResizeMode::Bicubic => {
                // No clamp here: torch keeps the negative coordinate for cubic
                // and clamps the four *indices* instead.
                let source = half_pixel - 0.5;
                let lower = source.floor();
                let fraction = source - lower;
                for tap in 0..4 {
                    let index = lower as isize - 1 + tap as isize;
                    self.index
                        .push(index.clamp(0, in_len as isize - 1) as usize);
                    self.weight.push(cubic(fraction + 1.0 - tap as f64));
                }
            }
            ResizeMode::Area => {
                let start = out_index * in_len / out_len;
                let end = ((out_index + 1) * in_len).div_ceil(out_len);
                let weight = 1.0 / (end - start) as f64;
                for index in start..end {
                    self.index.push(index);
                    self.weight.push(weight);
                }
            }
        }
    }

    /// The current taps, paired with their weights.
    fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.index.iter().copied().zip(self.weight.iter().copied())
    }
}

/// Resize a 1-D spatial tensor `[N, C, L]` to `[N, C, new_L]`.
///
/// # Panics
/// If `input` is not rank-3 or `new_l` is zero.
pub fn resize_1d<T, B>(
    input: &Tensor<T, B>,
    new_l: usize,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "resize_1d: expected [N,C,L] tensor");
    let scale = shape[2] as f64 / new_l as f64;
    resize_1d_with_scale(input, new_l, scale, mode)
}

/// [`resize_1d`] with an explicit sampling step of `scale` source units per
/// output position — pass `1 / scale_factor` to reproduce
/// `torch.nn.functional.interpolate(..., scale_factor=..)` for a fractional
/// factor, where torch keeps the requested factor instead of deriving it from
/// the rounded output size.
///
/// # Panics
/// If `input` is not rank-3, `new_l` is zero, or `scale` is not positive and
/// finite.
pub fn resize_1d_with_scale<T, B>(
    input: &Tensor<T, B>,
    new_l: usize,
    scale: f64,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "resize_1d: expected [N,C,L] tensor");
    assert!(new_l > 0, "resize_1d: new length must be non-zero");
    assert!(
        scale > 0.0 && scale.is_finite(),
        "resize_1d: scale must be positive and finite, got {scale}"
    );
    let (n, c, l) = (shape[0], shape[1], shape[2]);

    let input = input.to_contiguous()?;
    let source = input.as_slice();
    let mut output = vec![T::ZERO; n * c * new_l];
    let mut taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * l;
        for (out_index, slot) in output[plane * new_l..(plane + 1) * new_l]
            .iter_mut()
            .enumerate()
        {
            taps.fill(out_index, l, new_l, mode, scale);
            let mut value = T::ZERO;
            for (index, weight) in taps.iter() {
                value += source[base + index] * T::from_f64(weight);
            }
            *slot = value;
        }
    }

    Tensor::from_slice_on(vec![n, c, new_l], &output, &B::default())
}

/// Resize a 2-D spatial tensor `[N, C, H, W]` to `[N, C, new_H, new_W]`.
///
/// # Panics
/// If `input` is not rank-4, or `new_h`/`new_w` is zero.
pub fn resize_2d<T, B>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "resize_2d: expected [N,C,H,W] tensor");
    assert!(
        new_h > 0 && new_w > 0,
        "resize_2d: new extents must be non-zero"
    );
    let (h, w) = (shape[2], shape[3]);
    resize_2d_with_scale(
        input,
        new_h,
        new_w,
        h as f64 / new_h as f64,
        w as f64 / new_w as f64,
        mode,
    )
}

/// [`resize_2d`] with explicit per-axis sampling steps — see
/// [`resize_1d_with_scale`] for why a `scale_factor` caller must not derive
/// them from the output size.
///
/// # Panics
/// If `input` is not rank-4, `new_h`/`new_w` is zero, or a scale is not
/// positive and finite.
pub fn resize_2d_with_scale<T, B>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "resize_2d: expected [N,C,H,W] tensor");
    assert!(
        new_h > 0 && new_w > 0,
        "resize_2d: new extents must be non-zero"
    );
    assert!(
        scale_h > 0.0 && scale_h.is_finite() && scale_w > 0.0 && scale_w.is_finite(),
        "resize_2d: scales must be positive and finite, got {scale_h} and {scale_w}"
    );
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    let input = input.to_contiguous()?;
    let source = input.as_slice();
    let plane_in = h * w;
    let plane_out = new_h * new_w;
    let mut output = vec![T::ZERO; n * c * plane_out];
    let mut y_taps = TapTable::default();
    let mut x_taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * plane_in;
        for out_y in 0..new_h {
            y_taps.fill(out_y, h, new_h, mode, scale_h);
            for out_x in 0..new_w {
                x_taps.fill(out_x, w, new_w, mode, scale_w);
                let mut value = T::ZERO;
                for (y, y_weight) in y_taps.iter() {
                    for (x, x_weight) in x_taps.iter() {
                        value += source[base + y * w + x] * T::from_f64(y_weight * x_weight);
                    }
                }
                output[plane * plane_out + out_y * new_w + out_x] = value;
            }
        }
    }

    Tensor::from_slice_on(vec![n, c, new_h, new_w], &output, &B::default())
}

/// Gradient of [`resize_1d`] with respect to its input.
///
/// `input_shape` is the shape of the resized input (`[N, C, L]`) and
/// `grad_output` the `[N, C, new_L]` upstream gradient.
///
/// # Panics
/// If `input_shape` is not rank-3 or `grad_output` does not match the shape the
/// forward pass would have produced.
pub fn resize_1d_backward<T, B>(
    input_shape: &[usize],
    new_l: usize,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        3,
        "resize_1d_backward: expected [N,C,L] input shape"
    );
    let scale = input_shape[2] as f64 / new_l as f64;
    resize_1d_backward_with_scale(input_shape, new_l, scale, mode, grad_output)
}

/// [`resize_1d_backward`] with the forward pass's explicit sampling step, so
/// the scatter uses the same taps the forward pass applied.
pub fn resize_1d_backward_with_scale<T, B>(
    input_shape: &[usize],
    new_l: usize,
    scale: f64,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        3,
        "resize_1d_backward: expected [N,C,L] input shape"
    );
    let (n, c, l) = (input_shape[0], input_shape[1], input_shape[2]);
    let expected = [n, c, new_l];
    assert_eq!(
        grad_output.shape(),
        expected,
        "resize_1d_backward: gradient shape must match the resized output"
    );

    let grad_output = grad_output.to_contiguous()?;
    let upstream = grad_output.as_slice();
    let mut gradient = vec![T::ZERO; n * c * l];
    let mut taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * l;
        for out_index in 0..new_l {
            taps.fill(out_index, l, new_l, mode, scale);
            let g = upstream[plane * new_l + out_index];
            for (index, weight) in taps.iter() {
                gradient[base + index] += g * T::from_f64(weight);
            }
        }
    }

    Tensor::from_slice_on(input_shape.to_vec(), &gradient, &B::default())
}

/// Gradient of [`resize_2d`] with respect to its input.
///
/// `input_shape` is the shape of the resized input (`[N, C, H, W]`) and
/// `grad_output` the `[N, C, new_H, new_W]` upstream gradient.
///
/// # Panics
/// If `input_shape` is not rank-4 or `grad_output` does not match the shape the
/// forward pass would have produced.
pub fn resize_2d_backward<T, B>(
    input_shape: &[usize],
    new_h: usize,
    new_w: usize,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        4,
        "resize_2d_backward: expected [N,C,H,W] input shape"
    );
    let (h, w) = (input_shape[2], input_shape[3]);
    resize_2d_backward_with_scale(
        input_shape,
        new_h,
        new_w,
        h as f64 / new_h as f64,
        w as f64 / new_w as f64,
        mode,
        grad_output,
    )
}

/// [`resize_2d_backward`] with the forward pass's explicit sampling steps.
pub fn resize_2d_backward_with_scale<T, B>(
    input_shape: &[usize],
    new_h: usize,
    new_w: usize,
    scale_h: f64,
    scale_w: f64,
    mode: ResizeMode,
    grad_output: &Tensor<T, B>,
) -> Result<Tensor<T, B>, B::Error>
where
    T: Float + FloatElement,
    B: Backend,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        input_shape.len(),
        4,
        "resize_2d_backward: expected [N,C,H,W] input shape"
    );
    let (n, c, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let expected = [n, c, new_h, new_w];
    assert_eq!(
        grad_output.shape(),
        expected,
        "resize_2d_backward: gradient shape must match the resized output"
    );

    let grad_output = grad_output.to_contiguous()?;
    let upstream = grad_output.as_slice();
    let plane_in = h * w;
    let plane_out = new_h * new_w;
    let mut gradient = vec![T::ZERO; n * c * plane_in];
    let mut y_taps = TapTable::default();
    let mut x_taps = TapTable::default();

    for plane in 0..n * c {
        let base = plane * plane_in;
        for out_y in 0..new_h {
            y_taps.fill(out_y, h, new_h, mode, scale_h);
            for out_x in 0..new_w {
                x_taps.fill(out_x, w, new_w, mode, scale_w);
                let g = upstream[plane * plane_out + out_y * new_w + out_x];
                for (y, y_weight) in y_taps.iter() {
                    for (x, x_weight) in x_taps.iter() {
                        gradient[base + y * w + x] += g * T::from_f64(y_weight * x_weight);
                    }
                }
            }
        }
    }

    Tensor::from_slice_on(input_shape.to_vec(), &gradient, &B::default())
}
