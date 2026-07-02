use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Abstract interface for compile-time specialized unary autograd operations.
pub trait UnaryAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Send + Sync {
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B>;

    /// Compute input gradient: computes the derivative and scales by grad_out.
    ///
    /// Accepts both the input tensor `x` and output tensor `y` to allow optimized
    /// derivative computation using the output values where mathematically feasible.
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B>;
}

/// Autograd node for a generic unary activation operation.
pub struct UnaryNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved input tensor for backward computation.
    pub a_tensor: Tensor<T, B>,
    /// Saved output tensor for backward computation.
    pub out_tensor: Tensor<T, B>,
    /// Zero-sized phantom to bind the op type parameter.
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: UnaryAutogradOp<T, B>> BackwardNode<T, B>
    for UnaryNode<T, B, Op>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        Op::OP_NAME
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mask = Op::backward(grad_out, &self.a_tensor, &self.out_tensor, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &mask, &backend);
        }
    }
}

/// Generic, monomorphized activation wrapper that builds the autograd node.
#[inline]
pub fn unary_op<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    Op: UnaryAutogradOp<T, B> + 'static,
>(
    a: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, &backend);
    let requires_grad = crate::grad_mode::should_track_var(a);
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
        let inputs = vec![a.clone()];
        let a_tensor = a.tensor.clone();
        let out_t = out_tensor.clone();
        let node: UnaryNode<T, B, Op> = UnaryNode {
            output_grad,
            inputs,
            a_tensor,
            out_tensor: out_t,
            _phantom: std::marker::PhantomData,
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

// ── Leaf modules ──
/// Extended activation family (Hardtanh, Hardsigmoid, Hardswish, Hardshrink,
/// Softshrink, Softsign, Threshold, Celu). See `ext.rs` for subgradient
/// contracts at kink points.
pub mod ext;
/// GELU activation forward/backward nodes.
pub mod gelu;
/// Mathematical unary ops (abs, floor, round, sign, sqrt, etc.).
pub mod math;
/// ReLU-family activations (ReLU, LeakyReLU, ELU).
pub mod relu;
/// Sigmoid activation forward/backward.
pub mod sigmoid;
/// SiLU-family activations (SiLU, Mish, Softplus).
pub mod silu;
/// Tanh activation forward/backward.
pub mod tanh_act;
/// Trigonometric and exponential ops (sin, cos, exp, log).
pub mod trig;

// ── Re-exports ──
pub use gelu::{gelu, gelu_tanh, GeluOp, GeluTanhOp};
pub use math::{
    abs, ceil, clamp, floor, neg, pow, recip, round, sign, sqrt, trunc, AbsOp, CeilOp, ClampNode,
    FloorOp, NegOp, PowNode, RecipOp, RoundOp, SignOp, SqrtOp, TruncOp,
};
pub use relu::{elu, leaky_relu, prelu, relu, EluOp, ReluOp};
pub use sigmoid::{sigmoid, SigmoidOp};
pub use silu::{mish, silu, softplus, MishOp, SiluOp, SoftplusOp};
pub use tanh_act::{tanh, TanhOp};
pub use trig::{cos, erf, exp, log, sin, CosOp, ErfOp, ExpOp, LogOp, SinOp};
// Extended-family re-exports (G-037).
pub use ext::{
    celu, hardshrink, hardsigmoid, hardswish, hardtanh, pack_pairs, softshrink, softsign,
    threshold, HardsigmoidOp, HardswishOp, SoftsignOp,
};
