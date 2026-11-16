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

mod backward;
mod forward;
mod taps;

pub use backward::{
    resize_1d_backward, resize_1d_backward_with_scale, resize_2d_backward,
    resize_2d_backward_with_scale,
};
pub use forward::{resize_1d, resize_1d_with_scale, resize_2d, resize_2d_with_scale};
pub use taps::ResizeMode;
