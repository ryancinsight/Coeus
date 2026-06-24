// ── Interpolate — spatial resize / upsample / downsample ──
//
// Supported modes:
//   `nearest`  — map each output pixel to the nearest input pixel (floor)
//   `bilinear` — bilinear interpolation (4-neighbour weighted average)
//
// Input conventions (matching PyTorch and Burn):
//   1-D: `[N, C, L]`         → output `[N, C, new_L]`
//   2-D: `[N, C, H, W]`      → output `[N, C, new_H, new_W]`
//
// Forward-only (no autograd in this implementation).  For training, wrap the
// result in a `Var` with `requires_grad = false`.

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;

/// Resize mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolateMode {
    /// Nearest-neighbour — zero-cost conceptually, no blending.
    Nearest,
    /// Bilinear — 4-neighbour weighted average; only valid for 2-D inputs.
    Bilinear,
}

/// Resize a 1-D spatial tensor `[N, C, L]` to `[N, C, new_L]`.
///
/// Supported modes: [`InterpolateMode::Nearest`].
pub fn interpolate_1d<T: Float, B: coeus_core::Backend>(
    input: &Tensor<T, B>,
    new_l: usize,
    mode: InterpolateMode,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 3, "interpolate_1d: expected [N,C,L] tensor");
    let (n, c, l) = (shape[0], shape[1], shape[2]);

    let in_cont = input.to_contiguous();
    let in_s = in_cont.as_slice();

    let out_numel = n * c * new_l;
    let mut out = vec![T::zero(); out_numel];

    for bi in 0..n {
        for ci in 0..c {
            for xi in 0..new_l {
                let val = match mode {
                    InterpolateMode::Nearest => {
                        let src_x = ((xi as f64 + 0.5) * l as f64 / new_l as f64) as usize;
                        let src_x = src_x.min(l - 1);
                        in_s[bi * c * l + ci * l + src_x]
                    }
                    InterpolateMode::Bilinear => {
                        // 1-D bilinear = linear interpolation
                        let frac = (xi as f64 + 0.5) * l as f64 / new_l as f64 - 0.5;
                        let x0 = (frac.floor() as isize).max(0) as usize;
                        let x1 = (x0 + 1).min(l - 1);
                        let w1 = T::from_f64(frac - frac.floor());
                        let w0 = T::from_f64(1.0) - w1;
                        let v0 = in_s[bi * c * l + ci * l + x0];
                        let v1 = in_s[bi * c * l + ci * l + x1];
                        v0 * w0 + v1 * w1
                    }
                };
                out[bi * c * new_l + ci * new_l + xi] = val;
            }
        }
    }

    Tensor::from_slice(vec![n, c, new_l], &out)
}

/// Resize a 2-D spatial tensor `[N, C, H, W]` to `[N, C, new_H, new_W]`.
///
/// Supported modes: [`InterpolateMode::Nearest`] and [`InterpolateMode::Bilinear`].
pub fn interpolate_2d<T: Float, B: coeus_core::Backend>(
    input: &Tensor<T, B>,
    new_h: usize,
    new_w: usize,
    mode: InterpolateMode,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "interpolate_2d: expected [N,C,H,W] tensor");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    let in_cont = input.to_contiguous();
    let in_s = in_cont.as_slice();

    let mut out = vec![T::zero(); n * c * new_h * new_w];

    for bi in 0..n {
        for ci in 0..c {
            for yi in 0..new_h {
                for xi in 0..new_w {
                    let val = match mode {
                        InterpolateMode::Nearest => {
                            let sy = ((yi as f64 + 0.5) * h as f64 / new_h as f64) as usize;
                            let sx = ((xi as f64 + 0.5) * w as f64 / new_w as f64) as usize;
                            let sy = sy.min(h - 1);
                            let sx = sx.min(w - 1);
                            in_s[bi * c * h * w + ci * h * w + sy * w + sx]
                        }
                        InterpolateMode::Bilinear => {
                            // Align-half-pixel convention (same as PyTorch align_corners=False)
                            let fy = (yi as f64 + 0.5) * h as f64 / new_h as f64 - 0.5;
                            let fx = (xi as f64 + 0.5) * w as f64 / new_w as f64 - 0.5;
                            let y0 = (fy.floor() as isize).max(0) as usize;
                            let x0 = (fx.floor() as isize).max(0) as usize;
                            let y1 = (y0 + 1).min(h - 1);
                            let x1 = (x0 + 1).min(w - 1);
                            let wy = T::from_f64(fy - fy.floor());
                            let wx = T::from_f64(fx - fx.floor());
                            let wy0 = T::from_f64(1.0) - wy;
                            let wx0 = T::from_f64(1.0) - wx;
                            let base = bi * c * h * w + ci * h * w;
                            in_s[base + y0 * w + x0] * wy0 * wx0
                                + in_s[base + y0 * w + x1] * wy0 * wx
                                + in_s[base + y1 * w + x0] * wy * wx0
                                + in_s[base + y1 * w + x1] * wy * wx
                        }
                    };
                    out[bi * c * new_h * new_w + ci * new_h * new_w + yi * new_w + xi] = val;
                }
            }
        }
    }

    Tensor::from_slice(vec![n, c, new_h, new_w], &out)
}
