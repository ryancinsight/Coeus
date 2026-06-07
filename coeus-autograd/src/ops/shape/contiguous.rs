use std::sync::{Arc, Mutex};
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use crate::node::BackwardNode;
use crate::var::Var;

pub struct ContiguousNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ContiguousNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "contiguous"
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
        if let Some(Some(ref g)) = input_grads.first() {
            let mut gl = g.lock().unwrap();
            coeus_ops::add_assign(&mut gl, grad_out, &backend);
        }
    }
}

/// Tracked contiguous operation. Forces a copy to contiguous layout.
#[inline]
pub fn contiguous<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
) -> Var<T, B> {
    if x.tensor.is_contiguous() {
        return x.clone();
    }
    let backend = B::default();
    let out_tensor = x.tensor.to_contiguous();

    let requires_grad = x.grad.is_some();
    if !requires_grad {
        return Var::new(out_tensor, false);
    }

    let output_grad = Arc::new(Mutex::new(Tensor::zeros_on(out_tensor.shape_cloned(), &backend)));
    let grad = Some(output_grad.clone());

    let node = ContiguousNode {
        output_grad,
        inputs: vec![x.clone()],
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
