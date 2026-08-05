// ── Binary kernel ──
// Generic element-wise binary kernel with broadcasting.

use crate::backend_ops::{BinaryOp, ElementwiseOps};
use coeus_core::{BackendError, Scalar};
use coeus_tensor::broadcast::broadcast_shapes;
use coeus_tensor::Tensor;

/// Element-wise binary operation with broadcasting.
///
/// Uses `Tensor::alloc_on` (no zero-init) because every output element is
/// unconditionally overwritten by the broadcast kernel.
#[inline]
pub fn elementwise_binary<T: Scalar, B: ElementwiseOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    backend: &B,
    op: BinaryOp,
) -> Result<Tensor<T, B>, B::Error> {
    let out_shape = broadcast_shapes(a.shape(), b.shape()).ok_or_else(|| {
        B::Error::from(BackendError::IncompatibleBroadcast {
            operation: "elementwise_binary",
            from: a.shape().to_vec(),
            to: b.shape().to_vec(),
        })
    })?;

    let mut out: Tensor<T, B> = Tensor::alloc_on(out_shape.clone(), backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_binary(
        op,
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Apply element-wise binary operation to `a` and `b`, writing result to `out`.
#[inline]
pub fn elementwise_binary_to<T: Scalar, B: ElementwiseOps<T>>(
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
    op: BinaryOp,
) -> Result<(), B::Error> {
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_binary(
        op,
        a.storage(),
        a.layout(),
        b.storage(),
        b.layout(),
        out_storage,
        out_layout,
    )
}
