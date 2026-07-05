use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

#[inline]
pub(crate) fn avg_pool2d<T: Scalar, B: Backend>(
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
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c = input_layout.shape()[1];
    let h = input_layout.shape()[2];
    let w = input_layout.shape()[3];
    let h_out = output_layout.shape()[2];
    let w_out = output_layout.shape()[3];
    let out_numel = n * c * h_out * w_out;

    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice();

    let input_ptr = Ptr(input_slice.as_ptr());
    let output_ptr = MutPtr(output_slice.as_mut_ptr());

    let input_layout = input_layout.clone();
    let output_layout = output_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    backend.parallel_for(0, out_numel, move |i| {
        let ow = i % w_out;
        let temp1 = i / w_out;
        let oh = temp1 % h_out;
        let temp2 = temp1 / h_out;
        let ci = temp2 % c;
        let ni = temp2 / c;

        let mut sum = T::zero();
        let mut count = 0usize;

        for ikh in 0..k_size {
            let h_in = oh as isize * stride_s + ikh as isize * dil_s - pad_s;
            if h_in >= 0 && (h_in as usize) < h {
                for ikw in 0..k_size {
                    let w_in = ow as isize * stride_s + ikw as isize * dil_s - pad_s;
                    if w_in >= 0 && (w_in as usize) < w {
                        let input_idx =
                            input_layout.physical_index(&[ni, ci, h_in as usize, w_in as usize]);
                        let val = unsafe { input_ptr.read(input_idx) };
                        sum += val;
                        count += 1;
                    }
                }
            }
        }

        let mean = if count > 0 {
            sum / T::from_f64(count as f64)
        } else {
            T::zero()
        };

        let output_idx = output_layout.physical_index(&[ni, ci, oh, ow]);
        unsafe {
            output_ptr.write(output_idx, mean);
        }
    });
}

#[inline]
pub(crate) fn avg_pool2d_backward<T: Scalar, B: Backend>(
    backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut B::DeviceBuffer<T>,
    grad_input_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = grad_input_layout.shape()[0];
    let c = grad_input_layout.shape()[1];
    let h = grad_input_layout.shape()[2];
    let w = grad_input_layout.shape()[3];
    let h_out = grad_out_layout.shape()[2];
    let w_out = grad_out_layout.shape()[3];
    let numel_in = n * c * h * w;

    let go_slice = grad_out.as_slice();
    let gi_slice = grad_input.as_mut_slice();

    let go_ptr = Ptr(go_slice.as_ptr());
    let gi_ptr = MutPtr(gi_slice.as_mut_ptr());

    let go_layout = grad_out_layout.clone();
    let gi_layout = grad_input_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    backend.parallel_for(0, numel_in, move |i| {
        let wi = i % w;
        let temp1 = i / w;
        let hi = temp1 % h;
        let temp2 = temp1 / h;
        let ci = temp2 % c;
        let ni = temp2 / c;

        let mut sum = T::zero();

        for ikh in 0..k_size {
            let numer_h = hi as isize + pad_s - ikh as isize * dil_s;
            if numer_h >= 0 && numer_h % stride_s == 0 {
                let oh = (numer_h / stride_s) as usize;
                if oh < h_out {
                    for ikw in 0..k_size {
                        let numer_w = wi as isize + pad_s - ikw as isize * dil_s;
                        if numer_w >= 0 && numer_w % stride_s == 0 {
                            let ow = (numer_w / stride_s) as usize;
                            if ow < w_out {
                                let mut count = 0usize;
                                for jkh in 0..k_size {
                                    let h_in =
                                        oh as isize * stride_s + jkh as isize * dil_s - pad_s;
                                    if h_in >= 0 && (h_in as usize) < h {
                                        for jkw in 0..k_size {
                                            let w_in = ow as isize * stride_s
                                                + jkw as isize * dil_s
                                                - pad_s;
                                            if w_in >= 0 && (w_in as usize) < w {
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                                if count > 0 {
                                    let go_idx = go_layout.physical_index(&[ni, ci, oh, ow]);
                                    let gval = unsafe { go_ptr.read(go_idx) };
                                    sum += gval / T::from_f64(count as f64);
                                }
                            }
                        }
                    }
                }
            }
        }

        let gi_idx = gi_layout.physical_index(&[ni, ci, hi, wi]);
        unsafe {
            let old = gi_ptr.read(gi_idx);
            gi_ptr.write(gi_idx, old + sum);
        }
    });
}

#[inline]
pub(crate) fn avg_pool3d<T: Scalar, B: Backend>(
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
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c = input_layout.shape()[1];
    let d = input_layout.shape()[2];
    let h = input_layout.shape()[3];
    let w = input_layout.shape()[4];
    let d_out = output_layout.shape()[2];
    let h_out = output_layout.shape()[3];
    let w_out = output_layout.shape()[4];
    let out_numel = n * c * d_out * h_out * w_out;

    let input_slice = input.as_slice();
    let output_slice = output.as_mut_slice();

    let input_ptr = Ptr(input_slice.as_ptr());
    let output_ptr = MutPtr(output_slice.as_mut_ptr());

    let input_layout = input_layout.clone();
    let output_layout = output_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    backend.parallel_for(0, out_numel, move |i| {
        let ow = i % w_out;
        let temp1 = i / w_out;
        let oh = temp1 % h_out;
        let temp2 = temp1 / h_out;
        let od = temp2 % d_out;
        let temp3 = temp2 / d_out;
        let ci = temp3 % c;
        let ni = temp3 / c;

        let mut sum = T::zero();
        let mut count = 0usize;

        for ikd in 0..k_size {
            let d_in = od as isize * stride_s + ikd as isize * dil_s - pad_s;
            if d_in >= 0 && (d_in as usize) < d {
                for ikh in 0..k_size {
                    let h_in = oh as isize * stride_s + ikh as isize * dil_s - pad_s;
                    if h_in >= 0 && (h_in as usize) < h {
                        for ikw in 0..k_size {
                            let w_in = ow as isize * stride_s + ikw as isize * dil_s - pad_s;
                            if w_in >= 0 && (w_in as usize) < w {
                                let input_idx = input_layout.physical_index(&[
                                    ni,
                                    ci,
                                    d_in as usize,
                                    h_in as usize,
                                    w_in as usize,
                                ]);
                                let val = unsafe { input_ptr.read(input_idx) };
                                sum += val;
                                count += 1;
                            }
                        }
                    }
                }
            }
        }

        let mean = if count > 0 {
            sum / T::from_f64(count as f64)
        } else {
            T::zero()
        };

        let output_idx = output_layout.physical_index(&[ni, ci, od, oh, ow]);
        unsafe {
            output_ptr.write(output_idx, mean);
        }
    });
}

#[inline]
pub(crate) fn avg_pool3d_backward<T: Scalar, B: Backend>(
    backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut B::DeviceBuffer<T>,
    grad_input_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let n = grad_input_layout.shape()[0];
    let c = grad_input_layout.shape()[1];
    let d = grad_input_layout.shape()[2];
    let h = grad_input_layout.shape()[3];
    let w = grad_input_layout.shape()[4];
    let d_out = grad_out_layout.shape()[2];
    let h_out = grad_out_layout.shape()[3];
    let w_out = grad_out_layout.shape()[4];
    let numel_in = n * c * d * h * w;

    let go_slice = grad_out.as_slice();
    let gi_slice = grad_input.as_mut_slice();

    let go_ptr = Ptr(go_slice.as_ptr());
    let gi_ptr = MutPtr(gi_slice.as_mut_ptr());

    let go_layout = grad_out_layout.clone();
    let gi_layout = grad_input_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    backend.parallel_for(0, numel_in, move |i| {
        let wi = i % w;
        let temp1 = i / w;
        let hi = temp1 % h;
        let temp2 = temp1 / h;
        let di = temp2 % d;
        let temp3 = temp2 / d;
        let ci = temp3 % c;
        let ni = temp3 / c;

        let mut sum = T::zero();

        for ikd in 0..k_size {
            let numer_d = di as isize + pad_s - ikd as isize * dil_s;
            if numer_d >= 0 && numer_d % stride_s == 0 {
                let od = (numer_d / stride_s) as usize;
                if od < d_out {
                    for ikh in 0..k_size {
                        let numer_h = hi as isize + pad_s - ikh as isize * dil_s;
                        if numer_h >= 0 && numer_h % stride_s == 0 {
                            let oh = (numer_h / stride_s) as usize;
                            if oh < h_out {
                                for ikw in 0..k_size {
                                    let numer_w = wi as isize + pad_s - ikw as isize * dil_s;
                                    if numer_w >= 0 && numer_w % stride_s == 0 {
                                        let ow = (numer_w / stride_s) as usize;
                                        if ow < w_out {
                                            let mut count = 0usize;
                                            for jkd in 0..k_size {
                                                let d_in = od as isize * stride_s
                                                    + jkd as isize * dil_s
                                                    - pad_s;
                                                if d_in >= 0 && (d_in as usize) < d {
                                                    for jkh in 0..k_size {
                                                        let h_in = oh as isize * stride_s
                                                            + jkh as isize * dil_s
                                                            - pad_s;
                                                        if h_in >= 0 && (h_in as usize) < h {
                                                            for jkw in 0..k_size {
                                                                let w_in = ow as isize * stride_s
                                                                    + jkw as isize * dil_s
                                                                    - pad_s;
                                                                if w_in >= 0 && (w_in as usize) < w
                                                                {
                                                                    count += 1;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            if count > 0 {
                                                let go_idx =
                                                    go_layout.physical_index(&[ni, ci, od, oh, ow]);
                                                let gval = unsafe { go_ptr.read(go_idx) };
                                                sum += gval / T::from_f64(count as f64);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gi_idx = gi_layout.physical_index(&[ni, ci, di, hi, wi]);
        unsafe {
            let old = gi_ptr.read(gi_idx);
            gi_ptr.write(gi_idx, old + sum);
        }
    });
}
