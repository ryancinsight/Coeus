use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::{Arc, Mutex};

pub struct SoftmaxNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub y_clone: Tensor<T, B>,
    pub dim_u: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SoftmaxNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "softmax"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<Mutex<Tensor<T, B>>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<Mutex<Tensor<T, B>>>>]) {
        let backend = B::default();
        if let Some(Some(ref g_in)) = input_grads.get(0) {
            let gy = coeus_ops::mul(grad_out, &self.y_clone, &backend);
            let sum_gy = coeus_ops::sum_axis(&gy, self.dim_u, &backend);
            let mut dx = coeus_ops::sub(grad_out, &sum_gy, &backend);
            coeus_ops::mul_assign(&mut dx, &self.y_clone, &backend);
            let mut gl = g_in.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &dx, &backend);
        }
    }
}

/// Tracked Softmax.
pub fn softmax<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: isize,
) -> Var<T, B> {
    let ndim = input.tensor.ndim();
    let dim_u = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };
    assert!(
        dim_u < ndim,
        "softmax dim {dim} out of bounds for ndim={ndim}"
    );
    let backend = B::default();

    let max_t = coeus_ops::max_axis(&input.tensor, dim_u, &backend);
    let shift_x = coeus_ops::sub(&input.tensor, &max_t, &backend);
    let exp_x_t = coeus_ops::exp(&shift_x, &backend);
    let sum_t = coeus_ops::sum_axis(&exp_x_t, dim_u, &backend);
    let y_t = coeus_ops::div(&exp_x_t, &sum_t, &backend);

    let requires_grad = input.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            y_t.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![input.clone()];
        let y_clone = y_t.clone();

        let node = SoftmaxNode {
            output_grad,
            inputs,
            y_clone,
            dim_u,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: y_t,
        grad,
        creator,
    }
}
