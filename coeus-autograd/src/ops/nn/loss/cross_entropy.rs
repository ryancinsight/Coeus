use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for cross-entropy loss.
pub struct CrossEntropyLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Target class indices for each sample.
    pub targets: Vec<usize>,
    /// Softmax probabilities stored as `Vec<T>` — no f64 widening.
    pub probs: Vec<T>,
    /// Batch size.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for CrossEntropyLossNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "cross_entropy_loss"
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
            let temp_grad;
            let grad_out_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            let mut host_grad = [T::zero()];
            backend.copy_to_host(grad_out_cont.storage(), &mut host_grad);
            // Scale in T precision — no widening to f64
            let n_t = T::from_f64(self.n as f64);
            let grad_out_val = host_grad[0];
            let scale = grad_out_val / n_t;

            let mut d_logits = vec![T::zero(); self.n * self.c];
            for i in 0..self.n {
                let offset = i * self.c;
                let target_idx = self.targets[i];
                for j in 0..self.c {
                    let p = self.probs[offset + j];
                    let indicator = if j == target_idx { T::one() } else { T::zero() };
                    d_logits[offset + j] = (p - indicator) * scale;
                }
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &d_logits, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Cross-Entropy Loss.
/// Called from coeus-nn/src/loss.rs after host-side log-sum-exp computation.
/// `probs` must be `Vec<T>`, computed in T precision.
pub fn cross_entropy_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    targets: Vec<usize>,
    out_tensor: Tensor<T, B>,
    probs: Vec<T>,
    n: usize,
    c: usize,
) -> Var<T, B> {
    let backend = B::default();
    let requires_grad = crate::grad_mode::should_track_var(logits);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![logits.clone()];

        let node = CrossEntropyLossNode {
            output_grad,
            inputs,
            targets,
            probs,
            n,
            c,
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
