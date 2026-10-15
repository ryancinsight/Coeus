use super::super::{unary_op, UnaryAutogradOp};
use crate::var::Var;
use coeus_core::{Float, FloatElement};
use coeus_tensor::Tensor;

// `d/dx √x = 1 / (2√x) = grad_out / (2·y)` where `y = √x` (stored forward output).

unary_autograd!({Float + FloatElement} SqrtOp, "sqrt", sqrt, |g, _x, y, b| {
    let two = Tensor::full_on(y.shape(), <T as FloatElement>::from_f64(2.0), b)?;
    let denom = coeus_ops::mul(y, &two, b)?;
    coeus_ops::div(g, &denom, b)
});
