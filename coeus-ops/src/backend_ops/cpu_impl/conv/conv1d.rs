use super::brand_mut_slice;
use crate::ptr::{MutPtr, Ptr};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use melinoe::brand_scope;
use melinoe::sync::{partition_for_each_with, PartitionPlan};

#[inline]
pub(crate) fn conv1d<T: Scalar, B: Backend>(
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
    let l = input_layout.shape()[2];
    let c_out = weight_layout.shape()[0];
    let k = weight_layout.shape()[2];
    let l_out = output_layout.shape()[2];
    let out_numel = n * c_out * l_out;
    if out_numel == 0 {
        return;
    }

    let input_slice = input.as_slice();
    let weight_slice = weight.as_slice();
    let output_slice = output.as_mut_slice();
    let bias_slice = bias.map(|b| b.as_slice());

    let input_layout = input_layout.clone();
    let weight_layout = weight_layout.clone();
    let output_layout = output_layout.clone();
    let bias_layout = bias.map(|_| Layout::new([c_out].into()));

    let pad_s = padding as isize;
    let stride_s = stride as isize;
    let dil_s = dilation as isize;

    let has_canonical_contiguous_output = k <= l && stride > 0 && l_out == (l - k) / stride + 1;

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
        let out_rows = n * c_out;
        let output_region = &mut output_slice[output_offset..output_offset + out_numel];
        let row_kernel = move |row: usize, row_out: &mut [T]| {
            let oc = row % c_out;
            let ni = row / c_out;
            for (ol, slot) in row_out.iter_mut().enumerate() {
                let mut sum = T::zero();
                for ic in 0..c_in {
                    let input_start = input_offset + (ni * c_in + ic) * l + ol * stride;
                    let weight_start = weight_offset + (oc * c_in + ic) * k;
                    // SAFETY: the contiguous fast path proves the logical window
                    // `[input_start, input_start + k)` stays inside the backing
                    // input slice for each output position.
                    let input_window = &input_slice[input_start..input_start + k];
                    // SAFETY: row-major contiguous weights store each
                    // `(out_channel, in_channel)` kernel as one `k`-element run.
                    let weight_window = &weight_slice[weight_start..weight_start + k];
                    sum += T::dot_slice(input_window, weight_window);
                }
                if let Some(bs) = bias_slice {
                    sum += bs[oc];
                }
                *slot = sum;
            }
        };

        if backend.num_threads() <= 1 || out_rows <= 1 {
            for (row, row_out) in output_region.chunks_mut(l_out).enumerate() {
                row_kernel(row, row_out);
            }
        } else {
            // ── Atlas contention guard ────────────────────────────────────
            // `partition_for_each_with` shards the region across the available
            // worker pool; for tiny regions (e.g. conv1d on `2×8×128` with
            // ~2 K output cells split across ~16 threads → ~128 cells per
            // worker), the work-stealing-deque setup cost dominates the actual
            // compute. Bypass the partition driver when the per-thread share
            // would be too small to amortise scheduling; the provider-owned
            // benchmark records the resulting Sequential/Moirai behavior.
            //
            // Calibration: a single f32 conv-row kernel on `l_out` elements is
            // ~12·l_out flops / ~4·c_in·l_out + ~4·l_out bytes — a row of
            // `l_out = 126` cells at `c_in = 8` reads/writes ~5 KiB per row,
            // which fits in L1. We require each thread to own at least 4 such
            // rows (≈ 512 cells, ≥ 20 KiB L1 footprint) before paying the
            // partition overhead.
            const MIN_ROWS_PER_THREAD: usize = 4;
            let n_threads = backend.num_threads();
            if out_rows < MIN_ROWS_PER_THREAD * n_threads {
                for (row, row_out) in output_region.chunks_mut(l_out).enumerate() {
                    row_kernel(row, row_out);
                }
            } else {
                brand_scope(|_token| {
                    // SAFETY: `output_region` is a fresh exclusive borrow that lives
                    // entirely within this brand scope; `partition_for_each_with`
                    // then splits it into disjoint row-sized shards.
                    let output_cells = unsafe { brand_mut_slice(output_region) };
                    partition_for_each_with(
                        output_cells,
                        PartitionPlan::chunk_size(l_out),
                        |start, mut shard| {
                            row_kernel(start / l_out, shard.as_mut_slice());
                        },
                    );
                });
            }
        }
        return;
    }

    let input_ptr = Ptr(input_slice.as_ptr());
    let weight_ptr = Ptr(weight_slice.as_ptr());
    let output_ptr = MutPtr(output_slice.as_mut_ptr());
    let bias_ptr = bias_slice.map(|bs| Ptr(bs.as_ptr()));

    backend.parallel_for(0, out_numel, move |i| {
        let ol = i % l_out;
        let temp = i / l_out;
        let oc = temp % c_out;
        let ni = temp / c_out;

        let mut sum = T::zero();
        for ic in 0..c_in {
            for ik in 0..k {
                let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
                if l_in >= 0 && (l_in as usize) < l {
                    let input_idx = input_layout.physical_index(&[ni, ic, l_in as usize]);
                    let weight_idx = weight_layout.physical_index(&[oc, ic, ik]);
                    let ival = unsafe { input_ptr.read(input_idx) };
                    let wval = unsafe { weight_ptr.read(weight_idx) };
                    sum += ival * wval;
                }
            }
        }
        if let Some(ref bp) = bias_ptr {
            let bias_idx = bias_layout.as_ref().unwrap().physical_index(&[oc]);
            sum += unsafe { bp.read(bias_idx) };
        }
        let output_idx = output_layout.physical_index(&[ni, oc, ol]);
        unsafe {
            output_ptr.write(output_idx, sum);
        }
    });
}

#[inline]
pub(crate) fn conv1d_backward<T: Scalar, B: Backend>(
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
    let l = input_layout.shape()[2];
    let c_out = weight_layout.shape()[0];
    let k = weight_layout.shape()[2];
    let l_out = grad_out_layout.shape()[2];

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

        let numel_in = n * c_in * l;
        backend.parallel_for(0, numel_in, move |i| {
            let li = i % l;
            let temp = i / l;
            let ic = temp % c_in;
            let ni = temp / c_in;

            let mut sum = T::zero();
            for oc in 0..c_out {
                for ik in 0..k {
                    let numer = li as isize + pad_s - ik as isize * dil_s;
                    if numer >= 0 && numer % stride_s == 0 {
                        let ol = (numer / stride_s) as usize;
                        if ol < l_out {
                            let go_idx = go_layout.physical_index(&[ni, oc, ol]);
                            let w_idx = w_layout.physical_index(&[oc, ic, ik]);
                            let gval = unsafe { go_ptr.read(go_idx) };
                            let wval = unsafe { w_ptr.read(w_idx) };
                            sum += gval * wval;
                        }
                    }
                }
            }
            let gi_idx = gi_layout.physical_index(&[ni, ic, li]);
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

        let numel_w = c_out * c_in * k;
        let can_dot_grad_weight = padding == 0
            && stride == 1
            && dilation == 1
            && k <= l
            && l_out == l - k + 1
            && go_layout.is_contiguous()
            && input_layout.is_contiguous()
            && gw_layout.is_contiguous();

        if can_dot_grad_weight {
            let go_offset = go_layout.offset();
            let input_offset = input_layout.offset();
            let gw_offset = gw_layout.offset();
            backend.parallel_for(0, numel_w, move |i| {
                let ik = i % k;
                let temp = i / k;
                let ic = temp % c_in;
                let oc = temp / c_in;

                let mut sum = T::zero();
                for ni in 0..n {
                    let go_start = go_offset + (ni * c_out + oc) * l_out;
                    let input_start = input_offset + (ni * c_in + ic) * l + ik;
                    // SAFETY: the contiguous guard proves canonical NCL row-major
                    // layouts, `k <= l`, and `l_out == l - k + 1`, so both
                    // windows have length `l_out` and remain inside storage.
                    let go_window = unsafe { go_ptr.slice(go_start, l_out) };
                    let input_window = unsafe { input_ptr.slice(input_start, l_out) };
                    sum += T::dot_slice(go_window, input_window);
                }
                let gw_idx = gw_offset + (oc * c_in + ic) * k + ik;
                unsafe {
                    let old = gw_ptr.read(gw_idx);
                    gw_ptr.write(gw_idx, old + sum);
                }
            });
        } else {
            backend.parallel_for(0, numel_w, move |i| {
                let ik = i % k;
                let temp = i / k;
                let ic = temp % c_in;
                let oc = temp / c_in;

                let mut sum = T::zero();
                for ni in 0..n {
                    for ol in 0..l_out {
                        let l_in = ol as isize * stride_s + ik as isize * dil_s - pad_s;
                        if l_in >= 0 && (l_in as usize) < l {
                            let go_idx = go_layout.physical_index(&[ni, oc, ol]);
                            let input_idx = input_layout.physical_index(&[ni, ic, l_in as usize]);
                            let gval = unsafe { go_ptr.read(go_idx) };
                            let ival = unsafe { input_ptr.read(input_idx) };
                            sum += gval * ival;
                        }
                    }
                }
                let gw_idx = gw_layout.physical_index(&[oc, ic, ik]);
                unsafe {
                    let old = gw_ptr.read(gw_idx);
                    gw_ptr.write(gw_idx, old + sum);
                }
            });
        }
    }

    if let Some(gb) = grad_bias {
        let go_layout = go_layout.clone();
        let gb_slice = gb.as_mut_slice();
        let gb_ptr = MutPtr(gb_slice.as_mut_ptr());
        let gb_layout = Layout::new([c_out].into());

        backend.parallel_for(0, c_out, move |oc| {
            let mut sum = T::zero();
            for ni in 0..n {
                for ol in 0..l_out {
                    let go_idx = go_layout.physical_index(&[ni, oc, ol]);
                    let gval = unsafe { go_ptr.read(go_idx) };
                    sum += gval;
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
