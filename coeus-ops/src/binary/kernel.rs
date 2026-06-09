// ── Binary kernel ──
// Generic element-wise binary kernel with broadcasting.

use crate::backend_ops::{BackendOps, BinaryOp};
use coeus_core::Scalar;
use coeus_tensor::broadcast::broadcast_shapes;
use coeus_tensor::Tensor;

/// Element-wise binary operation with broadcasting.
#[inline]
pub fn elementwise_binary<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
    op: BinaryOp,
) -> Tensor<T, B> {
    let out_shape =
        broadcast_shapes(a.shape(), b.shape()).expect("Incompatible shapes for broadcasting");

    let mut out: Tensor<T, B> = Tensor::zeros_on(out_shape.clone(), backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_binary(
        op,
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        out_storage,
        out_layout,
    );

    out
}

/// Apply element-wise binary operation to `a` and `b`, writing result to `out`.
#[inline]
pub fn elementwise_binary_to<T: Scalar, B: BackendOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
    op: BinaryOp,
) {
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_binary(
        op,
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        out_storage,
        out_layout,
    );
}
