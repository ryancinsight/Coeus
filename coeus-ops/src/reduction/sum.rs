// ── Sum reduction ──

use coeus_core::Scalar;
use coeus_tensor::Tensor;
use crate::backend_ops::{BackendOps, ReductionOp};

/// Sum all elements.
#[inline]
pub fn sum<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    if a.numel() == 0 {
        return T::zero();
    }
    let reshaped = if a.is_contiguous() && a.layout().offset() == 0 {
        a.reshape([a.numel()])
    } else {
        let contiguous = a.to_contiguous_on(backend);
        contiguous.reshape([a.numel()])
    };
    let reduced = sum_axis(&reshaped, 0, backend);
    let mut host_scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut host_scalar);
    host_scalar[0]
}

/// Sum along a specific axis, reducing it to size 1.
#[inline]
pub fn sum_axis<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, axis: usize, backend: &B) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "sum_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Sum,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}

/// Maximum along a specific axis, reducing it to size 1.
#[inline]
pub fn max_axis<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, axis: usize, backend: &B) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "max_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Max,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}

/// Minimum along a specific axis, reducing it to size 1.
#[inline]
pub fn min_axis<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, axis: usize, backend: &B) -> Tensor<T, B> {
    assert!(axis < a.ndim(), "min_axis: axis {axis} out of bounds");

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::zeros_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Min,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    );

    out
}
