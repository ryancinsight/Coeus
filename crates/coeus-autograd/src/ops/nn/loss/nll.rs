use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for negative log-likelihood loss.
pub struct NllLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Target class indices for each sample.
    pub targets: Vec<usize>,
    /// Batch size.
    pub n: usize,
    /// Number of classes.
    pub c: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for NllLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "nll_loss"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.get(0) {
            let mut host_grad = [T::zero()];
            let temp_grad;
            let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
                grad_out
            } else {
                temp_grad = grad_out.to_contiguous_on(&backend);
                &temp_grad
            };
            backend.copy_to_host(grad_cont.storage(), &mut host_grad);
            let g_out = host_grad[0];
            let n_t = T::from_f64(self.n as f64);
            // Use T::zero() - x idiom for negation
            let neg_scale = T::zero() - (g_out / n_t);

            let mut d_log = vec![T::zero(); self.n * self.c];
            for i in 0..self.n {
                d_log[i * self.c + self.targets[i]] = neg_scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n, self.c], &d_log, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)
                .expect("autograd gradient accumulation");
        }
    }
}

/// Tracked Negative Log-Likelihood Loss.
/// log_probs: `[N, C]` (already log-probabilities), targets: `[N]` class indices.
pub fn nll_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B> {
    let backend = B::default();
    let shape = log_probs.tensor.shape();
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n);

    let cont;
    let log_raw = if log_probs.tensor.is_contiguous() && log_probs.tensor.layout().offset() == 0 {
        &log_probs.tensor
    } else {
        cont = log_probs.tensor.to_contiguous_on(&backend);
        &cont
    };

    let host: std::borrow::Cow<[T]> = if let Some(s) = log_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n * c])
    } else {
        let mut v = vec![T::zero(); n * c];
        backend.copy_to_host(log_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let mut loss_val = T::zero();
    for i in 0..n {
        // T::zero() - x for negation
        loss_val += T::zero() - host[i * c + targets[i]];
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(log_probs);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = NllLossNode {
            output_grad,
            inputs: vec![log_probs.clone()],
            targets: targets.to_vec(),
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
