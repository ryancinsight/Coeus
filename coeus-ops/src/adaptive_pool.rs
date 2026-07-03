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
//
// # Performance
// The contiguous fast-path uses raw slice arithmetic (`parallel_for` across
// (N,C) pairs) to avoid the per-element layout-index overhead of `get/set`.
// The non-contiguous slow-path falls back to `get/set`.

use crate::backend_ops::BackendOps;
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
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
///
/// # Memory
/// Output is allocated uninitialized (`alloc_on`); every `[n, c, o]` position
/// is written by the loop before the function returns.
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

    // alloc_on: every [ni, ci, oi] is written via set — no zero-init needed.
    let mut out = Tensor::alloc_on([n, c, output_size], backend);

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
///
/// # Memory
/// Output is `alloc_on` (uninitialized); every `[n, c, oi]` position is written
/// exactly once by the loop — no zero-initialization overhead.
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

    // alloc_on: every [ni, ci, oi] is written via set — no zero-init needed.
    let mut out = Tensor::alloc_on([n, c, output_size], backend);

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
/// Uses `parallel_for` over (N×C) pairs on contiguous inputs.
///
/// # Memory
/// Output is `alloc_on` (uninitialized); parallel_for writes every `(oh, ow)`
/// position — no zero-initialization overhead.
pub fn adaptive_avg_pool2d<T: Float, B: BackendOps<T> + Default + Backend>(
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

    // alloc_on: parallel_for writes every out position — no zero-init needed.
    let inp_cont;
    let inp = if input.is_contiguous() && input.layout().offset() == 0 {
        input
    } else {
        inp_cont = input.to_contiguous_on(backend);
        &inp_cont
    };

    use crate::ptr::{MutPtr, Ptr};
    let mut out = Tensor::alloc_on([n, c, out_h, out_w], backend);
    let inp_ptr = Ptr(inp.storage().as_slice().as_ptr());
    let out_ptr = MutPtr(out.storage_mut().as_mut_slice().as_mut_ptr());

    let nc = n * c;
    backend.parallel_for(0, nc, move |idx| {
        let ni = idx / c;
        let ci = idx % c;
        let inp_nc = ni * c * h * w + ci * h * w;
        let out_nc = ni * c * out_h * out_w + ci * out_h * out_w;
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
                        acc = acc + unsafe { inp_ptr.read(inp_nc + hi * w + wi) };
                    }
                }
                unsafe { out_ptr.write(out_nc + oh * out_w + ow, acc / count) };
            }
        }
    });

    out
}

// ── Adaptive Max Pool 2D ──────────────────────────────────────────────────────

/// Adaptive max pooling for `[N, C, H, W]` → `[N, C, out_h, out_w]`.
///
/// Equivalent to PyTorch `nn.AdaptiveMaxPool2d((out_h, out_w))`.
/// Uses `parallel_for` over (N×C) pairs on contiguous inputs.
///
/// # Memory
/// Output is `alloc_on` (uninitialized); parallel_for writes every position
/// exactly once — no zero-initialization overhead.
pub fn adaptive_max_pool2d<T: Float, B: BackendOps<T> + Default + Backend>(
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

    let inp_cont;
    let inp = if input.is_contiguous() && input.layout().offset() == 0 {
        input
    } else {
        inp_cont = input.to_contiguous_on(backend);
        &inp_cont
    };

    use crate::ptr::{MutPtr, Ptr};
    // alloc_on: parallel_for writes every out position — no zero-init needed.
    let mut out = Tensor::alloc_on([n, c, out_h, out_w], backend);
    let inp_ptr = Ptr(inp.storage().as_slice().as_ptr());
    let out_ptr = MutPtr(out.storage_mut().as_mut_slice().as_mut_ptr());

    let nc = n * c;
    backend.parallel_for(0, nc, move |idx| {
        let ni = idx / c;
        let ci = idx % c;
        let inp_nc = ni * c * h * w + ci * h * w;
        let out_nc = ni * c * out_h * out_w + ci * out_h * out_w;
        for oh in 0..out_h {
            let hs = region_start(oh, h, out_h);
            let he = region_end(oh, h, out_h);
            for ow in 0..out_w {
                let ws = region_start(ow, w, out_w);
                let we = region_end(ow, w, out_w);
                let mut max_val: Option<T> = None;
                for hi in hs..he {
                    for wi in ws..we {
                        let v = unsafe { inp_ptr.read(inp_nc + hi * w + wi) };
                        max_val = Some(max_val.map_or(v, |m| if v > m { v } else { m }));
                    }
                }
                unsafe { out_ptr.write(out_nc + oh * out_w + ow, max_val.unwrap_or(T::zero())) };
            }
        }
    });

    out
}
