// ── where_cond — conditional element-wise selection ──
//
// `where_cond(cond, on_true, on_false)` is the element-wise ternary:
//   out[i] = on_true[i]  if cond[i] != 0
//   out[i] = on_false[i] otherwise
//
// cond is treated as boolean: non-zero = true.
// All three tensors must have identical shapes.
//
// Composed from existing backend primitives (zero new kernels):
//   mask     = ReluGrad(cond)           — 1 where cond > 0, 0 elsewhere
//   neg_cond = -cond
//   neg_mask = ReluGrad(neg_cond)       — 1 where cond < 0, 0 elsewhere
//   any_mask = mask + neg_mask          — 1 where cond != 0
//   out      = on_true * any_mask + on_false * (1 - any_mask)

use crate::backend_ops::BackendOps;
use crate::backend_ops::UnaryOp;
use coeus_core::{BackendError, Float};
use coeus_tensor::Tensor;

/// Element-wise conditional select: `out[i] = if cond[i] != 0 { on_true[i] } else { on_false[i] }`.
///
/// All three tensors must have the **same shape**.  For broadcasting, pre-broadcast
/// `cond` before calling.
///
/// Composed purely from the existing backend primitives — no new GPU kernel required.
///
/// # Errors
/// Returns a typed shape-mismatch error if the three inputs do not have the
/// same shape, or propagates a backend dispatch failure.
#[inline]
pub fn where_cond<T: Float, B: BackendOps<T> + Default>(
    cond: &Tensor<T, B>,
    on_true: &Tensor<T, B>,
    on_false: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if cond.shape() != on_true.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "where_cond",
            lhs: cond.shape().to_vec(),
            rhs: on_true.shape().to_vec(),
        }));
    }
    if cond.shape() != on_false.shape() {
        return Err(B::Error::from(BackendError::ShapeMismatch {
            operation: "where_cond",
            lhs: cond.shape().to_vec(),
            rhs: on_false.shape().to_vec(),
        }));
    }

    // 1 where cond > 0, 0 elsewhere
    let mask_pos = crate::unary::elementwise_unary(cond, backend, UnaryOp::ReluGrad)?;
    // 1 where cond < 0, 0 elsewhere
    let cond_neg = crate::unary::elementwise_unary(cond, backend, UnaryOp::Neg)?;
    let mask_neg = crate::unary::elementwise_unary(&cond_neg, backend, UnaryOp::ReluGrad)?;
    // combined: 1 where cond != 0
    let any_mask = crate::binary::add(&mask_pos, &mask_neg, backend)?;

    let one = Tensor::full_on(any_mask.shape(), T::from_f64(1.0), backend)?;
    let inv_mask = crate::binary::sub(&one, &any_mask, backend)?;
    let true_part = crate::binary::mul(on_true, &any_mask, backend)?;
    let false_part = crate::binary::mul(on_false, &inv_mask, backend)?;
    Ok(crate::binary::add(&true_part, &false_part, backend)?)
}
