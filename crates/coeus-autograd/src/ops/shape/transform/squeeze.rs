use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct SqueezeNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub axis: Option<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SqueezeNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "squeeze"
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let unsqueezed_grad = if let Some(ax) = self.axis {
                grad_out.unsqueeze(ax)
            } else {
                let original_shape = self.inputs[0].tensor.shape_cloned();
                grad_out.reshape(original_shape)
            };
            let gl = g.write();
            coeus_ops::add_assign(gl, &unsqueezed_grad, &backend)?;
        }

        Ok(())
    }
}

/// Tracked squeeze operation.
#[inline]
pub fn squeeze<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    axis: Option<usize>,
) -> Result<Var<T, B>, B::Error> {
    let out_tensor = if let Some(ax) = axis {
        x.tensor.squeeze(ax)
    } else {
        x.tensor.squeeze_all()
    };

    let requires_grad = crate::grad_mode::should_track_var(x);
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )?));
    let grad = Some(output_grad.clone());

    let node = SqueezeNode {
        output_grad,
        inputs: vec![x.clone()],
        axis,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

pub struct UnsqueezeNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub axis: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for UnsqueezeNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "unsqueeze"
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let squeezed_grad = grad_out.squeeze(self.axis);
            let gl = g.write();
            coeus_ops::add_assign(gl, &squeezed_grad, &backend)?;
        }

        Ok(())
    }
}

/// Tracked unsqueeze operation.
#[inline]
pub fn unsqueeze<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    axis: usize,
) -> Result<Var<T, B>, B::Error> {
    let out_tensor = x.tensor.unsqueeze(axis);

    let requires_grad = crate::grad_mode::should_track_var(x);
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )?));
    let grad = Some(output_grad.clone());

    let node = UnsqueezeNode {
        output_grad,
        inputs: vec![x.clone()],
        axis,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
