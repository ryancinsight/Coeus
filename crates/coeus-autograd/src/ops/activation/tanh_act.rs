use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

unary_autograd!(TanhOp, "tanh", tanh, |g, _x, y, b| {
    super::backward_via_unary_derivative(g, y, b, coeus_ops::UnaryOp::TanhGrad)
});
