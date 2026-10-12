//! Zero-gradient ops.
//!
//! Sign, floor, ceil, round, trunc are non-differentiable (or have zero
//! gradient almost everywhere). Their backward returns a zero tensor.

use super::super::{unary_op, zero_unary_grad, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::{FloatOps, Scalar};
use coeus_tensor::Tensor;

unary_autograd!({Scalar + FloatOps} SignOp, "sign", sign, |g, _x, _y, b| {
    zero_unary_grad(g, b)
});
unary_autograd!({Scalar + FloatOps} FloorOp, "floor", floor, |g, _x, _y, b| {
    zero_unary_grad(g, b)
});
unary_autograd!({Scalar + FloatOps} CeilOp, "ceil", ceil, |g, _x, _y, b| {
    zero_unary_grad(g, b)
});
unary_autograd!({Scalar + FloatOps} RoundOp, "round", round, |g, _x, _y, b| {
    zero_unary_grad(g, b)
});
unary_autograd!({Scalar + FloatOps} TruncOp, "trunc", trunc, |g, _x, _y, b| {
    zero_unary_grad(g, b)
});
