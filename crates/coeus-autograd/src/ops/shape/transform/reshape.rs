use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::ops::shape::contiguous as make_contiguous;
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
            coeus_ops::add_assign(gl, &reshaped_grad, &backend)
                .expect("autograd gradient accumulation");
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
        let x_cont = make_contiguous(x);
        return reshape(&x_cont, shape);
    }
    let backend = B::default();
    let out_tensor = x.tensor.reshape(shape);

    let requires_grad = crate::grad_mode::should_track_var(x);
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

/// Collapse dimensions `start_dim..=end_dim` into a single dimension
/// (`torch.flatten`, `end_dim` inclusive). Differentiable via [`reshape`].
///
/// # Panics
/// If `start_dim > end_dim` or `end_dim` is out of range.
#[must_use]
pub fn flatten<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    start_dim: usize,
    end_dim: usize,
) -> Var<T, B> {
    let dims = x.tensor.shape();
    let ndim = dims.len();
    assert!(
        start_dim <= end_dim && end_dim < ndim,
        "flatten: dim range [{start_dim}, {end_dim}] invalid for rank {ndim}"
    );
    let mut new_shape: Vec<usize> = dims[..start_dim].to_vec();
    new_shape.push(dims[start_dim..=end_dim].iter().product());
    new_shape.extend_from_slice(&dims[end_dim + 1..]);
    reshape(x, new_shape)
}

#[cfg(test)]
mod flatten_tests {
    use super::*;
    use coeus_core::MoiraiBackend;

    #[test]
    fn flatten_collapses_dims_and_backprops() {
        // [2,3,4] flatten(1,2) -> [2,12], row-major values preserved.
        let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
        let x = Var::<f64, MoiraiBackend>::new(Tensor::from_slice([2, 3, 4], &data), true);
        let flat = flatten(&x, 1, 2);
        assert_eq!(flat.tensor.shape(), &[2, 12]);
        assert_eq!(flat.tensor.to_contiguous().as_slice(), data.as_slice());
        flat.backward();
        assert_eq!(x.grad().unwrap().shape(), &[2, 3, 4]);
    }
}
