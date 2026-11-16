//! Resize modes and the tap table that both forward and reverse mode read.

use coeus_core::FloatElement;
use eunomia::convert::IntegerTarget;

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
pub(super) struct TapTable {
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
    pub(super) fn fill(
        &mut self,
        out_index: usize,
        in_len: usize,
        out_len: usize,
        mode: ResizeMode,
        scale: f64,
    ) {
        self.index.clear();
        self.weight.clear();
        let half_pixel = (f64::from_count(out_index) + 0.5) * scale;
        match mode {
            ResizeMode::Nearest => {
                // `floor(i * scale)` — no `+ 0.5`. torch's `nearest` kernel
                // copies the cell the output cell *starts* in, so the two
                // agree on every integer ratio and diverge as soon as the
                // ratio is fractional.
                let source =
                    usize::from_truncated(f64::from_count(out_index) * scale).min(in_len - 1);
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
                let lower_index = usize::from_truncated(lower);
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
                let lower_index = isize::from_truncated(lower);
                let last_index =
                    isize::try_from(in_len).expect("invariant: an axis length fits isize") - 1;
                for tap in 0..4_usize {
                    let offset = isize::try_from(tap).expect("invariant: four taps fit isize");
                    let index = (lower_index - 1 + offset).clamp(0, last_index);
                    self.index.push(
                        usize::try_from(index).expect("invariant: a clamped index is non-negative"),
                    );
                    self.weight
                        .push(cubic(fraction + 1.0 - f64::from_count(tap)));
                }
            }
            ResizeMode::Area => {
                let start = out_index * in_len / out_len;
                let end = ((out_index + 1) * in_len).div_ceil(out_len);
                let weight = 1.0 / f64::from_count(end - start);
                for index in start..end {
                    self.index.push(index);
                    self.weight.push(weight);
                }
            }
        }
    }

    /// The current taps, paired with their weights.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.index.iter().copied().zip(self.weight.iter().copied())
    }
}
