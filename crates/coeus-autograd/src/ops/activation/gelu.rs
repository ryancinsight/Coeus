use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

unary_autograd!(GeluOp, "gelu", gelu, |g, x, _y, b| {
    let deriv = coeus_ops::elementwise_unary(x, b, coeus_ops::UnaryOp::GeluGrad)
        .expect("elementwise_unary");
    coeus_ops::mul(g, &deriv, b)
});

unary_autograd!(GeluTanhOp, "gelu_tanh", gelu_tanh, |g, x, _y, b| {
    let deriv = coeus_ops::elementwise_unary(x, b, coeus_ops::UnaryOp::GeluTanhGrad)
        .expect("elementwise_unary");
    coeus_ops::mul(g, &deriv, b)
});
