//! Tracked element-wise reciprocal.
//!
//! `d/dx [1/x] = −1/x² = −y²` where `y = 1/x` (stored forward output).

use super::super::{unary_op, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;

unary_autograd!({Scalar + FloatOps} RecipOp, "recip", recip, |g, _x, y, b| {
    let y_sq = coeus_ops::mul(y, y, b);
    let neg_y_sq = coeus_ops::neg(&y_sq, b);
    coeus_ops::mul(g, &neg_y_sq, b)
});
