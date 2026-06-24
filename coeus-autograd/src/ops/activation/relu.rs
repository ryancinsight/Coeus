use super::unary_op;
use super::UnaryAutogradOp;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST tag for ReLU autograd.
pub struct ReluOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ReluOp {
    const OP_NAME: &'static str = "relu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::relu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad);
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// Tracked ReLU activation.
#[must_use]
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ReluOp>(a)
}

/// Inline backward node for LeakyReLU.
struct LeakyReluNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    negative_slope: u64, // f64::to_bits
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LeakyReluNode<T, B> {
    fn op_name(&self) -> &'static str {
        "leaky_relu"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::LeakyReluGrad(self.negative_slope),
            );
            let mask = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &mask, &backend);
        }
    }
}

/// Tracked Leaky ReLU activation.
#[must_use]
#[inline]
pub fn leaky_relu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    negative_slope: f64,
) -> Var<T, B> {
    let backend = B::default();
    let slope_bits = f64::to_bits(negative_slope);
    let out_tensor = coeus_ops::leaky_relu(&a.tensor, &backend, negative_slope);
    let requires_grad = a.grad.is_some();

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = LeakyReluNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            negative_slope: slope_bits,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}

/// ZST tag for ELU autograd (alpha=1.0).
pub struct EluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for EluOp {
    const OP_NAME: &'static str = "elu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        // EluGrad takes the original input x and returns exp(x) or 1
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::EluGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked ELU activation.
#[must_use]
#[inline]
pub fn elu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, EluOp>(a)
}
