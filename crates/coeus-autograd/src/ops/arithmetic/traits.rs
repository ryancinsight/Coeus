use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Abstract interface for compile-time specialized binary autograd operations.
pub trait BinaryAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>: Send + Sync {
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B>;

    /// Compute input gradients backward.
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        a_shape: &Shape,
        b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    );
}

/// Autograd node for a generic binary operation.
pub struct BinaryNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: BinaryAutogradOp<T, B>>
{
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Saved first input tensor for backward computation.
    pub a_tensor: Tensor<T, B>,
    /// Saved second input tensor for backward computation.
    pub b_tensor: Tensor<T, B>,
    /// Shape of the first input, for broadcast gradient reduction.
    pub a_shape: Shape,
    /// Shape of the second input, for broadcast gradient reduction.
    pub b_shape: Shape,
    /// Zero-sized phantom to bind the op type parameter.
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: BinaryAutogradOp<T, B>>
    BackwardNode<T, B> for BinaryNode<T, B, Op>
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
        Op::backward(
            grad_out,
            &self.a_tensor,
            &self.b_tensor,
            &self.a_shape,
            &self.b_shape,
            input_grads,
            &backend,
        );
    }
}

/// Generic, monomorphized binary wrapper that builds the autograd node.
#[inline]
pub fn binary_op<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    Op: BinaryAutogradOp<T, B> + 'static,
>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, &b.tensor, &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(a) || crate::grad_mode::should_track_var(b);
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
        let inputs = vec![a.clone(), b.clone()];
        let a_shape: Shape = a.tensor.shape_cloned();
        let b_shape: Shape = b.tensor.shape_cloned();
        let a_tensor = a.tensor.clone();
        let b_tensor = b.tensor.clone();
        let node: BinaryNode<T, B, Op> = BinaryNode {
            output_grad,
            inputs,
            a_tensor,
            b_tensor,
            a_shape,
            b_shape,
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

/// Abstract interface for compile-time specialized reduction autograd operations.
pub trait ReductionAutogradOp<T: Scalar, B: coeus_ops::BackendOps<T> + Default>:
    Send + Sync
{
    /// Human-readable operation name for tracking.
    const OP_NAME: &'static str;

    /// Execute forward pass.
    fn forward(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Tensor<T, B>;

    /// Return optional scaling tensor for backward propagation.
    fn scaler(a: &Tensor<T, B>, param: Option<usize>, backend: &B) -> Option<Tensor<T, B>>;
}

/// Autograd node for reduction operations (sum, mean, norm, etc.).
pub struct ReductionNode<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    Op: ReductionAutogradOp<T, B>,
> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Shape of the input tensor, used to broadcast gradients on backward.
    pub a_shape: Shape,
    /// Optional scaling tensor applied during backward (e.g. for mean reduction).
    pub scaler_tensor: Option<Tensor<T, B>>,
    /// Zero-sized phantom to bind the op type parameter.
    pub _phantom: std::marker::PhantomData<Op>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, Op: ReductionAutogradOp<T, B>>
    BackwardNode<T, B> for ReductionNode<T, B, Op>
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
            let grad_to_broadcast = if let Some(ref scaler) = self.scaler_tensor {
                coeus_ops::mul(grad_out, scaler, &backend)
            } else {
                grad_out.clone()
            };
            let broadcasted = grad_to_broadcast.broadcast(self.a_shape.clone());
            let gl = g.write();
            coeus_ops::add_assign(gl, &broadcasted, &backend)
                .expect("autograd gradient accumulation");
        }
    }
}

/// Generic, monomorphized reduction wrapper that builds the autograd node.
#[inline]
pub fn reduction_op<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    Op: ReductionAutogradOp<T, B> + 'static,
>(
    a: &Var<T, B>,
    param: Option<usize>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = Op::forward(&a.tensor, param, &backend);
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
        let a_shape: Shape = a.tensor.shape_cloned();
        let scaler_tensor = Op::scaler(&a.tensor, param, &backend);
        let node: ReductionNode<T, B, Op> = ReductionNode {
            output_grad,
            inputs,
            a_shape,
            scaler_tensor,
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
