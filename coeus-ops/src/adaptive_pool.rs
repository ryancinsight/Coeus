// ── Adaptive pooling CPU kernels ──
//
// Implements region-based adaptive average and adaptive max pooling in 1D and 2D.
//
// For each output position `i` in `[0, out_size)`:
//   start_i = floor(i * in_size / out_size)
//   end_i   = ceil((i+1) * in_size / out_size)
//   window  = input[start_i..end_i]
//
// This matches PyTorch `nn.AdaptiveAvgPool1d` / `nn.AdaptiveAvgPool2d` /
// `nn.AdaptiveMaxPool1d` / `nn.AdaptiveMaxPool2d` for integer output sizes.

use crate::backend_ops::BackendOps;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;

// ── Region helpers ────────────────────────────────────────────────────────────

/// Start index (inclusive) for adaptive pool output position `i`.
#[inline]
fn region_start(i: usize, in_size: usize, out_size: usize) -> usize {
    (i * in_size) / out_size
}

/// End index (exclusive) for adaptive pool output position `i`.
#[inline]
fn region_end(i: usize, in_size: usize, out_size: usize) -> usize {
    ((i + 1) * in_size).div_ceil(out_size)
}

// ── Adaptive Avg Pool 1D ──────────────────────────────────────────────────────

/// Adaptive average pooling for `[N, C, L]` → `[N, C, output_size]`.
///
/// Equivalent to PyTorch `nn.AdaptiveAvgPool1d(output_size)`.
pub fn adaptive_avg_pool1d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    output_size: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(input.ndim(), 3, "adaptive_avg_pool1d expects [N, C, L]");
    let n = input.shape()[0];
    let c = input.shape()[1];
    let l = input.shape()[2];

    let mut out = Tensor::zeros_on([n, c, output_size], backend);

    for ni in 0..n {
        for ci in 0..c {
            for oi in 0..output_size {
                let start = region_start(oi, l, output_size);
                let end = region_end(oi, l, output_size);
                let count = T::from_f64((end - start) as f64);
                let mut acc = T::zero();
                for li in start..end {
                    acc = acc + input.get(&[ni, ci, li]);
                }
                out.set(&[ni, ci, oi], acc / count);
            }
        }
    }

    out
}

// ── Adaptive Max Pool 1D ──────────────────────────────────────────────────────

/// Adaptive max pooling for `[N, C, L]` → `[N, C, output_size]`.
///
/// Equivalent to PyTorch `nn.AdaptiveMaxPool1d(output_size)`.
pub fn adaptive_max_pool1d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    output_size: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(input.ndim(), 3, "adaptive_max_pool1d expects [N, C, L]");
    let n = input.shape()[0];
    let c = input.shape()[1];
    let l = input.shape()[2];

    let mut out = Tensor::zeros_on([n, c, output_size], backend);

    for ni in 0..n {
        for ci in 0..c {
            for oi in 0..output_size {
                let start = region_start(oi, l, output_size);
                let end = region_end(oi, l, output_size);
                let mut max_val: Option<T> = None;
                for li in start..end {
                    let v = input.get(&[ni, ci, li]);
                    max_val = Some(max_val.map_or(v, |m| if v > m { v } else { m }));
                }
                out.set(&[ni, ci, oi], max_val.unwrap_or(T::zero()));
            }
        }
    }

    out
}

// ── Adaptive Avg Pool 2D ──────────────────────────────────────────────────────

/// Adaptive average pooling for `[N, C, H, W]` → `[N, C, out_h, out_w]`.
///
/// Equivalent to PyTorch `nn.AdaptiveAvgPool2d((out_h, out_w))`.
pub fn adaptive_avg_pool2d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    out_h: usize,
    out_w: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(input.ndim(), 4, "adaptive_avg_pool2d expects [N, C, H, W]");
    let n = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];

    let mut out = Tensor::zeros_on([n, c, out_h, out_w], backend);

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                let hs = region_start(oh, h, out_h);
                let he = region_end(oh, h, out_h);
                for ow in 0..out_w {
                    let ws = region_start(ow, w, out_w);
                    let we = region_end(ow, w, out_w);
                    let count = T::from_f64(((he - hs) * (we - ws)) as f64);
                    let mut acc = T::zero();
                    for hi in hs..he {
                        for wi in ws..we {
                            acc = acc + input.get(&[ni, ci, hi, wi]);
                        }
                    }
                    out.set(&[ni, ci, oh, ow], acc / count);
                }
            }
        }
    }

    out
}

// ── Adaptive Max Pool 2D ──────────────────────────────────────────────────────

/// Adaptive max pooling for `[N, C, H, W]` → `[N, C, out_h, out_w]`.
///
/// Equivalent to PyTorch `nn.AdaptiveMaxPool2d((out_h, out_w))`.
pub fn adaptive_max_pool2d<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    out_h: usize,
    out_w: usize,
    backend: &B,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(input.ndim(), 4, "adaptive_max_pool2d expects [N, C, H, W]");
    let n = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];

    let mut out = Tensor::zeros_on([n, c, out_h, out_w], backend);

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                let hs = region_start(oh, h, out_h);
                let he = region_end(oh, h, out_h);
                for ow in 0..out_w {
                    let ws = region_start(ow, w, out_w);
                    let we = region_end(ow, w, out_w);
                    let mut max_val: Option<T> = None;
                    for hi in hs..he {
                        for wi in ws..we {
                            let v = input.get(&[ni, ci, hi, wi]);
                            max_val = Some(max_val.map_or(v, |m| if v > m { v } else { m }));
                        }
                    }
                    out.set(&[ni, ci, oh, ow], max_val.unwrap_or(T::zero()));
                }
            }
        }
    }

    out
}
