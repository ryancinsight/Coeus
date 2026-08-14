// ── Unfold / Fold CPU kernels (1D and 2D) ──
//
// Unfold 1D: [N, C, L] → [N, C*kernel_size, L_out]
// Fold   1D: [N, C*kernel_size, L_out] → [N, C, L]  (accumulate/sum overlap)
//
// Unfold 2D: [N, C, H, W] → [N, C*kH*kW, H_out*W_out]
// Fold   2D: [N, C*kH*kW, H_out*W_out] → [N, C, H, W]
//
// All kernels follow the PyTorch nn.Unfold / nn.Fold convention.

use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

pub(super) mod validation;

// ── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn out_dim(in_size: usize, kernel: usize, padding: usize, stride: usize, dilation: usize) -> usize {
    (in_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

// ── Unfold 1D ────────────────────────────────────────────────────────────────

/// Extract sliding windows from `[N, C, L]` into `[N, C*kernel_size, L_out]`.
#[inline]
pub(crate) fn unfold1d<T: Scalar, B: Backend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c = input_layout.shape()[1];
    let l = input_layout.shape()[2];
    let l_out = out_dim(l, kernel_size, padding, stride, dilation);
    let ck = c * kernel_size;
    let out_numel = n * ck * l_out;

    let input_ptr = Ptr(input.as_slice().as_ptr());
    let output_ptr = MutPtr(output.as_mut_slice().as_mut_ptr());
    let input_layout = input_layout.clone();
    let output_layout = output_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;

    backend.parallel_for(0, out_numel, move |idx| {
        // output[n, c*k + ki, lo]
        let lo = idx % l_out;
        let tmp = idx / l_out;
        let ck_idx = tmp % ck;
        let ni = tmp / ck;

        let ki = ck_idx % kernel_size;
        let ci = ck_idx / kernel_size;

        let l_in = lo as isize * stride_s + ki as isize * dil_s - pad_s;
        let val = if l_in >= 0 && (l_in as usize) < l {
            let src = input_layout.physical_index(&[ni, ci, l_in as usize]);
            unsafe { input_ptr.read(src) }
        } else {
            T::zero()
        };

        let dst = output_layout.physical_index(&[ni, ck_idx, lo]);
        unsafe { output_ptr.write(dst, val) };
    });
}

// ── Fold 1D ──────────────────────────────────────────────────────────────────

/// Accumulate `[N, C*kernel_size, L_out]` back into `[N, C, output_size]`.
/// Overlapping windows are summed (matches PyTorch nn.Fold).
#[inline]
pub(crate) fn fold1d<T: Scalar, B: Backend>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let ck = input_layout.shape()[1];
    let l_out = input_layout.shape()[2];
    let c = ck / kernel_size;

    // Zero output first.
    for v in output.as_mut_slice().iter_mut() {
        *v = T::zero();
    }

    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;

    for ni in 0..n {
        for ci in 0..c {
            for ki in 0..kernel_size {
                let ck_idx = ci * kernel_size + ki;
                for lo in 0..l_out {
                    let l_in = lo as isize * stride_s + ki as isize * dil_s - pad_s;
                    if l_in >= 0 && (l_in as usize) < output_size {
                        let src_idx = input_layout.physical_index(&[ni, ck_idx, lo]);
                        let dst_idx = output_layout.physical_index(&[ni, ci, l_in as usize]);
                        output_slice[dst_idx] += input_slice[src_idx];
                    }
                }
            }
        }
    }

    let _ = output_slice; // satisfy borrow checker
}

// ── Unfold 2D ────────────────────────────────────────────────────────────────

/// Extract sliding windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
#[inline]
pub(crate) fn unfold2d<T: Scalar, B: Backend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c = input_layout.shape()[1];
    let h = input_layout.shape()[2];
    let w = input_layout.shape()[3];
    let h_out = out_dim(h, kernel_h, padding_h, stride_h, dilation_h);
    let w_out = out_dim(w, kernel_w, padding_w, stride_w, dilation_w);
    let l_out = h_out * w_out;
    let ckk = c * kernel_h * kernel_w;
    let out_numel = n * ckk * l_out;

    let input_ptr = Ptr(input.as_slice().as_ptr());
    let output_ptr = MutPtr(output.as_mut_slice().as_mut_ptr());
    let input_layout = input_layout.clone();
    let output_layout = output_layout.clone();

    let pad_h_s = padding_h as isize;
    let pad_w_s = padding_w as isize;
    let str_h_s = stride_h as isize;
    let str_w_s = stride_w as isize;
    let dil_h_s = dilation_h as isize;
    let dil_w_s = dilation_w as isize;

    backend.parallel_for(0, out_numel, move |idx| {
        // output[n, (c*kh + khi)*kw + kwi, ho*w_out + wo]
        let lo = idx % l_out;
        let tmp = idx / l_out;
        let ckk_idx = tmp % ckk;
        let ni = tmp / ckk;

        let kwi = ckk_idx % kernel_w;
        let tmp2 = ckk_idx / kernel_w;
        let khi = tmp2 % kernel_h;
        let ci = tmp2 / kernel_h;

        let ho = lo / w_out;
        let wo = lo % w_out;

        let h_in = ho as isize * str_h_s + khi as isize * dil_h_s - pad_h_s;
        let w_in = wo as isize * str_w_s + kwi as isize * dil_w_s - pad_w_s;

        let val = if h_in >= 0 && (h_in as usize) < h && w_in >= 0 && (w_in as usize) < w {
            let src = input_layout.physical_index(&[ni, ci, h_in as usize, w_in as usize]);
            unsafe { input_ptr.read(src) }
        } else {
            T::zero()
        };

        let dst = output_layout.physical_index(&[ni, ckk_idx, lo]);
        unsafe { output_ptr.write(dst, val) };
    });
}

// ── Fold 2D ──────────────────────────────────────────────────────────────────

/// Accumulate `[N, C*kH*kW, H_out*W_out]` back into `[N, C, output_h, output_w]`.
#[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
#[inline]
pub(crate) fn fold2d<T: Scalar, B: Backend>(
    _backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let ckk = input_layout.shape()[1];
    let l_out = input_layout.shape()[2];
    let h_out = {
        let w_out = out_dim(output_w, kernel_w, padding_w, stride_w, dilation_w);
        l_out / w_out
    };
    let w_out = l_out / h_out;
    let c = ckk / (kernel_h * kernel_w);

    // Zero output first.
    for v in output.as_mut_slice().iter_mut() {
        *v = T::zero();
    }

    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice();

    let pad_h_s = padding_h as isize;
    let pad_w_s = padding_w as isize;
    let str_h_s = stride_h as isize;
    let str_w_s = stride_w as isize;
    let dil_h_s = dilation_h as isize;
    let dil_w_s = dilation_w as isize;

    for ni in 0..n {
        for ci in 0..c {
            for khi in 0..kernel_h {
                for kwi in 0..kernel_w {
                    let ckk_idx = (ci * kernel_h + khi) * kernel_w + kwi;
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let lo = ho * w_out + wo;
                            let h_in = ho as isize * str_h_s + khi as isize * dil_h_s - pad_h_s;
                            let w_in = wo as isize * str_w_s + kwi as isize * dil_w_s - pad_w_s;
                            if h_in >= 0
                                && (h_in as usize) < output_h
                                && w_in >= 0
                                && (w_in as usize) < output_w
                            {
                                let src = input_layout.physical_index(&[ni, ckk_idx, lo]);
                                let dst = output_layout.physical_index(&[
                                    ni,
                                    ci,
                                    h_in as usize,
                                    w_in as usize,
                                ]);
                                output_slice[dst] += input_slice[src];
                            }
                        }
                    }
                }
            }
        }
    }
}
