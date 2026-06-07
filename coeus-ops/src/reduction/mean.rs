// ── Mean reduction ──

use coeus_core::Scalar;
use coeus_tensor::Tensor;
use crate::backend_ops::BackendOps;

/// Mean of all elements.
#[inline]
pub fn mean<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    let total = super::sum::sum(a, backend);
    let count = T::from_f64(a.numel() as f64);
    total / count
}

/// Mean along a specific axis.
#[inline]
pub fn mean_axis<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, axis: usize, backend: &B) -> Tensor<T, B> {
    let mut summed = super::sum::sum_axis(a, axis, backend);
    let axis_len = a.shape()[axis];
    let count = T::from_f64(axis_len as f64);
    let counts = Tensor::from_slice_on([1], &[count], backend);
    crate::binary::div_assign(&mut summed, &counts, backend);
    summed
}
