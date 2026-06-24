use super::contiguous;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct ReshapeNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub original_shape: Shape,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ReshapeNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "reshape"
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
        if let Some(Some(ref g)) = input_grads.first() {
            let reshaped_grad = grad_out.reshape(self.original_shape.clone());
            let gl = g.write();
            coeus_ops::add_assign(gl, &reshaped_grad, &backend);
        }
    }
}

/// Tracked reshape operation. Automatically makes non-contiguous inputs contiguous.
#[inline]
pub fn reshape<T: Scalar, B: coeus_ops::BackendOps<T> + Default, S: Into<Shape>>(
    x: &Var<T, B>,
    shape: S,
) -> Var<T, B> {
    let shape = shape.into();
    if !x.tensor.is_contiguous() {
        let x_cont = contiguous::contiguous(x);
        return reshape(&x_cont, shape);
    }
    let backend = B::default();
    let out_tensor = x.tensor.reshape(shape);

    let requires_grad = x.grad.is_some();
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )));
    let grad = Some(output_grad.clone());

    let node = ReshapeNode {
        output_grad,
        inputs: vec![x.clone()],
        original_shape: x.tensor.shape_cloned(),
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
