// ── Embedding autograd operator ──

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for tracking embedding lookup operations.
pub struct EmbeddingNode<T: Scalar, I: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Index tensor used for the embedding lookup.
    pub indices: Tensor<I, B>,
    /// Number of embedding rows in the weight table.
    pub num_embeddings: usize,
    /// Optional padding index whose gradient is zeroed.
    pub padding_idx: Option<usize>,
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
        if let Some(Some(ref gw)) = input_grads.get(0) {
            let gw_update = coeus_ops::embedding_backward_with_padding_idx(
                grad_out,
                &self.indices,
                self.num_embeddings,
                self.padding_idx,
                &backend,
            )?;
            let gl = gw.write();
            coeus_ops::add_assign(gl, &gw_update, &backend)?;
        }

        Ok(())
    }
}

/// Tracked embedding lookup operation.
pub fn embedding<T: Scalar, I: Scalar + 'static, B: coeus_ops::BackendOps<T> + Default>(
    weight: &Var<T, B>,
    indices: &Tensor<I, B>,
) -> Result<Var<T, B>, B::Error> {
    embedding_with_padding_idx(weight, indices, None)
}

/// Tracked embedding lookup operation with an optional padding row whose
/// gradient is forced to zero during backward.
pub fn embedding_with_padding_idx<
    T: Scalar,
    I: Scalar + 'static,
    B: coeus_ops::BackendOps<T> + Default,
>(
    weight: &Var<T, B>,
    indices: &Tensor<I, B>,
    padding_idx: Option<usize>,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::embedding(&weight.tensor, indices, &backend)?;
    let requires_grad = crate::grad_mode::should_track_var(weight);

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if let Some(ref output_grad) = grad {
        let inputs = vec![weight.clone()];
        let indices_clone = indices.clone();
        let num_embeddings = weight.tensor.shape()[0];

        let node = EmbeddingNode {
            output_grad: output_grad.clone(),
            inputs,
            indices: indices_clone,
            num_embeddings,
            padding_idx,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
