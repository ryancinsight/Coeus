// ── Mean reduction ──

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Mean of all elements.
#[inline]
pub fn mean<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    let total = super::sum::sum(a, backend);
    let count = T::from_f64(a.numel() as f64);
    total / count
}

/// Mean along a specific axis.
#[inline]
pub fn mean_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "mean_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Mean,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}
