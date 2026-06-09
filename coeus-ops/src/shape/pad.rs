// ── Constant padding ──
// Pads a tensor with a constant value along each dimension.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_tensor::Tensor;

/// Pad `x` with `value` along each dimension.
///
/// `pads` is a list of `(before, after)` pairs, one per dimension.
/// Zero padding on all sides is equivalent to a clone.
///
/// # Panics
/// - `pads.len() != x.ndim()`.
#[inline]
pub fn pad<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    pads: &[(usize, usize)],
    value: T,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = x.ndim();
    assert_eq!(pads.len(), ndim, "pad: pads.len() must equal tensor ndim");

    // Fast path: no padding anywhere.
    if pads.iter().all(|&(b, a)| b == 0 && a == 0) {
        return x.clone();
    }

    let backend = B::default();
    let mut out_shape = x.shape_cloned();
    for d in 0..ndim {
        out_shape[d] += pads[d].0 + pads[d].1;
    }

    let mut out = Tensor::full_on(out_shape.clone(), value, &backend);
    let out_strides = Layout::new(out_shape.clone()).strides_cloned();

    let x_cont = x.to_contiguous_on(&backend);
    let x_slice = x_cont.as_slice();
    let x_shape = x_cont.shape();
    let x_strides = Layout::new(x_cont.shape_cloned()).strides_cloned();

    let numel_x = x_cont.numel();
    let out_data = out.as_mut_slice();

    for flat_in in 0..numel_x {
        let mut rem = flat_in;
        let src_phys = flat_in; // source is contiguous
        let mut dst_phys = 0usize;
        for d in (0..ndim).rev() {
            let coord = rem % x_shape[d];
            rem /= x_shape[d];
            let dst_coord = coord + pads[d].0;
            dst_phys += dst_coord * out_strides[d];
        }
        let _ = (src_phys, x_strides[0]); // suppress unused
        out_data[dst_phys] = x_slice[flat_in];
    }
    out
}
