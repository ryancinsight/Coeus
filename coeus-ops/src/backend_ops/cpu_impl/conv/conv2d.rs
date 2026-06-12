use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

#[inline]
pub(crate) fn conv2d<T: Scalar, B: Backend>(
    backend: &B,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    weight: &B::DeviceBuffer<T>,
    weight_layout: &Layout,
    bias: Option<&B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c_in = input_layout.shape()[1];
    let h = input_layout.shape()[2];
    let w = input_layout.shape()[3];
    let c_out = weight_layout.shape()[0];
    let kh = weight_layout.shape()[2];
    let kw = weight_layout.shape()[3];
    let h_out = output_layout.shape()[2];
    let w_out = output_layout.shape()[3];
    let out_numel = n * c_out * h_out * w_out;

    let input_slice = input.as_slice();
    let weight_slice = weight.as_slice();
    let output_slice = output.as_mut_slice();

    let input_ptr = Ptr(input_slice.as_ptr());
    let weight_ptr = Ptr(weight_slice.as_ptr());
    let output_ptr = MutPtr(output_slice.as_mut_ptr());

    let bias_ptr = bias.map(|b| Ptr(b.as_slice().as_ptr()));

    let input_layout = input_layout.clone();
    let weight_layout = weight_layout.clone();
    let output_layout = output_layout.clone();
    let bias_layout = bias.map(|_| Layout::new([c_out].into()));

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;

    let has_canonical_contiguous_output = kh <= h
        && kw <= w
        && stride > 0
        && h_out == (h - kh) / stride + 1
        && w_out == (w - kw) / stride + 1;

    if has_canonical_contiguous_output
        && padding == 0
        && dilation == 1
        && input_layout.is_contiguous()
        && weight_layout.is_contiguous()
        && output_layout.is_contiguous()
    {
        let input_offset = input_layout.offset();
        let weight_offset = weight_layout.offset();
        let output_offset = output_layout.offset();

        backend.parallel_for(0, out_numel, move |i| {
            let ow = i % w_out;
            let temp1 = i / w_out;
            let oh = temp1 % h_out;
            let temp2 = temp1 / h_out;
            let oc = temp2 % c_out;
            let ni = temp2 / c_out;

            let mut sum = T::zero();
            for ic in 0..c_in {
                for ikh in 0..kh {
                    let h_in = oh * stride + ikh;
                    let input_start =
                        input_offset + ((ni * c_in + ic) * h + h_in) * w + ow * stride;
                    let weight_start = weight_offset + ((oc * c_in + ic) * kh + ikh) * kw;
                    // SAFETY: the contiguous fast path is active only for
                    // row-major input/weight layouts with zero padding and
                    // unit dilation. `output_layout.shape()` constrains
                    // `ow * stride + kw <= w`, so both ranges are in bounds.
                    let input_window = unsafe { input_ptr.slice(input_start, kw) };
                    // SAFETY: row-major weight layout stores each kernel row
                    // as a contiguous `kw`-element run.
                    let weight_window = unsafe { weight_ptr.slice(weight_start, kw) };
                    sum = sum + T::dot_slice(input_window, weight_window);
                }
            }
            if let Some(ref bp) = bias_ptr {
                let bias_idx = bias_layout.as_ref().unwrap().physical_index(&[oc]);
                sum = sum + unsafe { bp.read(bias_idx) };
            }
            unsafe {
                output_ptr.write(output_offset + i, sum);
            }
        });
        return;
    }

    backend.parallel_for(0, out_numel, move |i| {
        let ow = i % w_out;
        let temp1 = i / w_out;
        let oh = temp1 % h_out;
        let temp2 = temp1 / h_out;
        let oc = temp2 % c_out;
        let ni = temp2 / c_out;

        let mut sum = T::zero();
        for ic in 0..c_in {
            for ikh in 0..kh {
                let h_in = oh as isize * stride_s + ikh as isize * dil_s - pad_s;
                if h_in >= 0 && (h_in as usize) < h {
                    for ikw in 0..kw {
                        let w_in = ow as isize * stride_s + ikw as isize * dil_s - pad_s;
                        if w_in >= 0 && (w_in as usize) < w {
                            let input_idx = input_layout.physical_index(&[
                                ni,
                                ic,
                                h_in as usize,
                                w_in as usize,
                            ]);
                            let weight_idx = weight_layout.physical_index(&[oc, ic, ikh, ikw]);
                            let ival = unsafe { input_ptr.read(input_idx) };
                            let wval = unsafe { weight_ptr.read(weight_idx) };
                            sum = sum + ival * wval;
                        }
                    }
                }
            }
        }
        if let Some(ref bp) = bias_ptr {
            let bias_idx = bias_layout.as_ref().unwrap().physical_index(&[oc]);
            sum = sum + unsafe { bp.read(bias_idx) };
        }
        let output_idx = output_layout.physical_index(&[ni, oc, oh, ow]);
        unsafe {
            output_ptr.write(output_idx, sum);
        }
    });
}

#[inline]
pub(crate) fn conv2d_backward<T: Scalar, B: Backend>(
    backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    weight: &B::DeviceBuffer<T>,
    weight_layout: &Layout,
    grad_input: Option<&mut B::DeviceBuffer<T>>,
    grad_input_layout: &Layout,
    grad_weight: Option<&mut B::DeviceBuffer<T>>,
    grad_weight_layout: &Layout,
    grad_bias: Option<&mut B::DeviceBuffer<T>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c_in = input_layout.shape()[1];
    let h = input_layout.shape()[2];
    let w = input_layout.shape()[3];
    let c_out = weight_layout.shape()[0];
    let kh = weight_layout.shape()[2];
    let kw = weight_layout.shape()[3];
    let h_out = grad_out_layout.shape()[2];
    let w_out = grad_out_layout.shape()[3];

    let go_slice = grad_out.as_slice();
    let go_ptr = Ptr(go_slice.as_ptr());
    let go_layout = grad_out_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;

    if let Some(gi) = grad_input {
        let go_layout = go_layout.clone();
        let gi_slice = gi.as_mut_slice();
        let gi_ptr = MutPtr(gi_slice.as_mut_ptr());
        let gi_layout = grad_input_layout.clone();

        let w_slice = weight.as_slice();
        let w_ptr = Ptr(w_slice.as_ptr());
        let w_layout = weight_layout.clone();

        let numel_in = n * c_in * h * w;
        backend.parallel_for(0, numel_in, move |i| {
            let wi = i % w;
            let temp1 = i / w;
            let hi = temp1 % h;
            let temp2 = temp1 / h;
            let ic = temp2 % c_in;
            let ni = temp2 / c_in;

            let mut sum = T::zero();
            for oc in 0..c_out {
                for ikh in 0..kh {
                    let numer_h = hi as isize + pad_s - ikh as isize * dil_s;
                    if numer_h >= 0 && numer_h % stride_s == 0 {
                        let oh = (numer_h / stride_s) as usize;
                        if oh < h_out {
                            for ikw in 0..kw {
                                let numer_w = wi as isize + pad_s - ikw as isize * dil_s;
                                if numer_w >= 0 && numer_w % stride_s == 0 {
                                    let ow = (numer_w / stride_s) as usize;
                                    if ow < w_out {
                                        let go_idx = go_layout.physical_index(&[ni, oc, oh, ow]);
                                        let w_idx = w_layout.physical_index(&[oc, ic, ikh, ikw]);
                                        let gval = unsafe { go_ptr.read(go_idx) };
                                        let wval = unsafe { w_ptr.read(w_idx) };
                                        sum = sum + gval * wval;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let gi_idx = gi_layout.physical_index(&[ni, ic, hi, wi]);
            unsafe {
                let old = gi_ptr.read(gi_idx);
                gi_ptr.write(gi_idx, old + sum);
            }
        });
    }

    if let Some(gw) = grad_weight {
        let go_layout = go_layout.clone();
        let gw_slice = gw.as_mut_slice();
        let gw_ptr = MutPtr(gw_slice.as_mut_ptr());
        let gw_layout = grad_weight_layout.clone();

        let input_slice = input.as_slice();
        let input_ptr = Ptr(input_slice.as_ptr());
        let input_layout = input_layout.clone();

        let numel_w = c_out * c_in * kh * kw;
        backend.parallel_for(0, numel_w, move |i| {
            let ikw = i % kw;
            let temp1 = i / kw;
            let ikh = temp1 % kh;
            let temp2 = temp1 / kh;
            let ic = temp2 % c_in;
            let oc = temp2 / c_in;

            let mut sum = T::zero();
            for ni in 0..n {
                for oh in 0..h_out {
                    let h_in = oh as isize * stride_s + ikh as isize * dil_s - pad_s;
                    if h_in >= 0 && (h_in as usize) < h {
                        for ow in 0..w_out {
                            let w_in = ow as isize * stride_s + ikw as isize * dil_s - pad_s;
                            if w_in >= 0 && (w_in as usize) < w {
                                let go_idx = go_layout.physical_index(&[ni, oc, oh, ow]);
                                let input_idx = input_layout.physical_index(&[
                                    ni,
                                    ic,
                                    h_in as usize,
                                    w_in as usize,
                                ]);
                                let gval = unsafe { go_ptr.read(go_idx) };
                                let ival = unsafe { input_ptr.read(input_idx) };
                                sum = sum + gval * ival;
                            }
                        }
                    }
                }
            }
            let gw_idx = gw_layout.physical_index(&[oc, ic, ikh, ikw]);
            unsafe {
                let old = gw_ptr.read(gw_idx);
                gw_ptr.write(gw_idx, old + sum);
            }
        });
    }

    if let Some(gb) = grad_bias {
        let go_layout = go_layout.clone();
        let gb_slice = gb.as_mut_slice();
        let gb_ptr = MutPtr(gb_slice.as_mut_ptr());
        let gb_layout = Layout::new([c_out].into());

        backend.parallel_for(0, c_out, move |oc| {
            let mut sum = T::zero();
            for ni in 0..n {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx = go_layout.physical_index(&[ni, oc, oh, ow]);
                        let gval = unsafe { go_ptr.read(go_idx) };
                        sum = sum + gval;
                    }
                }
            }
            let gb_idx = gb_layout.physical_index(&[oc]);
            unsafe {
                let old = gb_ptr.read(gb_idx);
                gb_ptr.write(gb_idx, old + sum);
            }
        });
    }
}
