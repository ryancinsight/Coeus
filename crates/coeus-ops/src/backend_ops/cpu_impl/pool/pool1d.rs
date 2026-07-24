// ── 1D pooling CPU kernels (max and avg, forward and backward) ──
//
// Input layout:  `[N, C, L]`
// Output layout: `[N, C, L_out]`
// L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1

use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};

// ── Max Pool 1D forward ──

#[inline]
pub(crate) fn max_pool1d<T: Scalar, B: Backend>(
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
    let l = input_layout.shape()[2];
    let l_out = output_layout.shape()[2];
    let out_numel = n * c * l_out;

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
        let ol = i % l_out;
        let temp = i / l_out;
        let ci = temp % c;
        let ni = temp / c;

        let mut max_val: Option<T> = None;

        for ik in 0..k_size {
            let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
            if l_in >= 0 && (l_in as usize) < l {
                let input_idx = input_layout.physical_index(&[ni, ci, l_in as usize]);
                let val = unsafe { input_ptr.read(input_idx) };
                max_val = Some(match max_val {
                    None => val,
                    Some(m) => {
                        if val > m {
                            val
                        } else {
                            m
                        }
                    }
                });
            }
        }

        let output_idx = output_layout.physical_index(&[ni, ci, ol]);
        unsafe {
            output_ptr.write(output_idx, max_val.unwrap_or(T::zero()));
        }
    });
}

// ── Max Pool 1D backward ──

#[inline]
pub(crate) fn max_pool1d_backward<T: Scalar, B: Backend>(
    backend: &B,
    grad_out: &B::DeviceBuffer<T>,
    grad_out_layout: &Layout,
    input: &B::DeviceBuffer<T>,
    input_layout: &Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &mut B::DeviceBuffer<T>,
    grad_input_layout: &Layout,
) where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let n = input_layout.shape()[0];
    let c = input_layout.shape()[1];
    let l = input_layout.shape()[2];
    let l_out = grad_out_layout.shape()[2];
    let out_numel = n * c * l_out;

    let input_slice = input.as_slice();
    let grad_out_slice = grad_out.as_slice();
    let grad_input_slice = grad_input.as_mut_slice();

    let input_ptr = Ptr(input_slice.as_ptr());
    let grad_out_ptr = Ptr(grad_out_slice.as_ptr());
    let grad_input_ptr = MutPtr(grad_input_slice.as_mut_ptr());

    let input_layout = input_layout.clone();
    let grad_out_layout = grad_out_layout.clone();
    let grad_input_layout = grad_input_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    // Sequential pass: for each output element, find argmax in window and accumulate.
    // Parallel version would need atomic adds; use sequential for correctness.
    for i in 0..out_numel {
        let ol = i % l_out;
        let temp = i / l_out;
        let ci = temp % c;
        let ni = temp / c;

        let mut max_val: Option<T> = None;
        let mut max_l_in: usize = 0;

        for ik in 0..k_size {
            let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
            if l_in >= 0 && (l_in as usize) < l {
                let idx = input_layout.physical_index(&[ni, ci, l_in as usize]);
                let val = unsafe { input_ptr.read(idx) };
                let better = match max_val {
                    None => true,
                    Some(m) => val > m,
                };
                if better {
                    max_val = Some(val);
                    max_l_in = l_in as usize;
                }
            }
        }

        if max_val.is_some() {
            let g_idx = grad_out_layout.physical_index(&[ni, ci, ol]);
            let g_val = unsafe { grad_out_ptr.read(g_idx) };
            let gi_idx = grad_input_layout.physical_index(&[ni, ci, max_l_in]);
            unsafe {
                let cur = grad_input_ptr.read(gi_idx);
                grad_input_ptr.write(gi_idx, cur + g_val);
            }
        }
    }
    let _ = backend; // backend unused in sequential backward
}

// ── Avg Pool 1D forward ──

#[inline]
pub(crate) fn avg_pool1d<T: Scalar, B: Backend>(
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
    let l = input_layout.shape()[2];
    let l_out = output_layout.shape()[2];
    let out_numel = n * c * l_out;

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
        let ol = i % l_out;
        let temp = i / l_out;
        let ci = temp % c;
        let ni = temp / c;

        let mut sum = T::zero();
        let mut count = 0usize;

        for ik in 0..k_size {
            let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
            if l_in >= 0 && (l_in as usize) < l {
                let input_idx = input_layout.physical_index(&[ni, ci, l_in as usize]);
                let val = unsafe { input_ptr.read(input_idx) };
                sum += val;
                count += 1;
            }
        }

        let mean = if count > 0 {
            sum / T::from_f64(count as f64)
        } else {
            T::zero()
        };

        let output_idx = output_layout.physical_index(&[ni, ci, ol]);
        unsafe {
            output_ptr.write(output_idx, mean);
        }
    });
}

// ── Avg Pool 1D backward ──

#[inline]
pub(crate) fn avg_pool1d_backward<T: Scalar, B: Backend>(
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
    let l = grad_input_layout.shape()[2];
    let l_out = grad_out_layout.shape()[2];
    let out_numel = n * c * l_out;

    let grad_out_slice = grad_out.as_slice();
    let grad_input_slice = grad_input.as_mut_slice();

    let grad_out_ptr = Ptr(grad_out_slice.as_ptr());
    let grad_input_ptr = MutPtr(grad_input_slice.as_mut_ptr());

    let grad_out_layout = grad_out_layout.clone();
    let grad_input_layout = grad_input_layout.clone();

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;
    let k_size = kernel_size;

    // Sequential backward: scatter 1/count gradient to each input element in the window.
    for i in 0..out_numel {
        let ol = i % l_out;
        let temp = i / l_out;
        let ci = temp % c;
        let ni = temp / c;

        // Count valid elements in the window.
        let mut count = 0usize;
        for ik in 0..k_size {
            let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
            if l_in >= 0 && (l_in as usize) < l {
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }

        let g_idx = grad_out_layout.physical_index(&[ni, ci, ol]);
        let g_val = unsafe { grad_out_ptr.read(g_idx) };
        let g_share = g_val / T::from_f64(count as f64);

        for ik in 0..k_size {
            let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
            if l_in >= 0 && (l_in as usize) < l {
                let gi_idx = grad_input_layout.physical_index(&[ni, ci, l_in as usize]);
                unsafe {
                    let cur = grad_input_ptr.read(gi_idx);
                    grad_input_ptr.write(gi_idx, cur + g_share);
                }
            }
        }
    }
    let _ = backend;
}
