use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

unary_autograd!(SiluOp, "silu", silu, |g, x, _y, b| {
    super::backward_via_unary_derivative(g, x, b, coeus_ops::UnaryOp::SiluGrad)
});

unary_autograd!(MishOp, "mish", mish, |g, x, _y, b| {
    super::backward_via_unary_derivative(g, x, b, coeus_ops::UnaryOp::MishGrad)
});

unary_autograd!(SoftplusOp, "softplus", softplus, |g, x, _y, b| {
    // SoftplusGrad = sigmoid(x).
    super::backward_via_unary_derivative(g, x, b, coeus_ops::UnaryOp::SoftplusGrad)
});
