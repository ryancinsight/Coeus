//! Tracked element-wise pairwise maximum / minimum (Burn `max_pair` /
//! `min_pair`, `torch.maximum` / `torch.minimum`).
//!
//! Both are expressed as scalar-free compositions of existing tracked ops via
//! the ReLU identities
//!   `maximum(a, b) = a + relu(b - a)`
//!   `minimum(a, b) = a - relu(a - b)`
//! so the autograd graph carries the gradient automatically and the ops run on
//! every backend with no new primitive. The gradient routes entirely to the
//! larger (resp. smaller) operand; on ties (`a == b`) it resolves to the first
//! argument `a`, since `relu'(0) = 0` in the tracked ReLU.

use crate::var::Var;
use crate::{add, relu, sub};
use coeus_core::Scalar;

/// Tracked element-wise maximum (`torch.maximum`, Burn `Tensor::max_pair`).
///
/// `maximum(a, b) = a + relu(b - a)`. Gradient flows to the larger operand;
/// ties resolve to `a`.
#[inline]
pub fn maximum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    let difference = sub(b, a)?;
    let positive = relu(&difference)?;
    add(a, &positive)
}

/// Tracked element-wise minimum (`torch.minimum`, Burn `Tensor::min_pair`).
///
/// `minimum(a, b) = a - relu(a - b)`. Gradient flows to the smaller operand;
/// ties resolve to `a`.
#[inline]
pub fn minimum<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    let difference = sub(a, b)?;
    let positive = relu(&difference)?;
    sub(a, &positive)
}
