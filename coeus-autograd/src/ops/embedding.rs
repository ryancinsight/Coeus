// ── Embedding autograd operator ──

use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Autograd node for tracking embedding lookup operations.
pub struct EmbeddingNode<T: Scalar, I: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<Mutex<Tensor<T, B>>>,
    pub inputs: Vec<Var<T, B>>,
    pub indices: Tensor<I, B>,
    pub num_embeddings: usize,
}

impl<T: Scalar, I: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for EmbeddingNode<T, I, B>
where
    I: Send + Sync + 'static,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "embedding"
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
        if let Some(Some(ref gw)) = input_grads.get(0) {
            let gw_update = coeus_ops::embedding_backward(
                grad_out,
                &self.indices,
                self.num_embeddings,
                &backend,
            );
            let mut gl = gw.lock().unwrap();
            coeus_ops::add_assign(&mut *gl, &gw_update, &backend);
        }
    }
}

/// Tracked embedding lookup operation.
pub fn embedding<T: Scalar, I: Scalar + 'static, B: coeus_ops::BackendOps<T> + Default>(
    weight: &Var<T, B>,
    indices: &Tensor<I, B>,
) -> Var<T, B> {
    let backend = B::default();
    let out_tensor = coeus_ops::embedding(&weight.tensor, indices, &backend);
    let requires_grad = weight.grad.is_some();

    let grad = if requires_grad {
        Some(Arc::new(Mutex::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![weight.clone()];
        let indices_clone = indices.clone();
        let num_embeddings = weight.tensor.shape()[0];

        let node = EmbeddingNode {
            output_grad,
            inputs,
            indices: indices_clone,
            num_embeddings,
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
