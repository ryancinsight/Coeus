use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for binary cross-entropy loss.
pub struct BinaryCrossEntropyNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Clamped prediction values in `[eps, 1-eps]`, stored as `Vec<T>`.
    pub probs: Vec<T>,
    /// Target values (0.0 or 1.0) stored as `Vec<T>`.
    pub targets: Vec<T>,
    /// Number of elements in the loss reduction.
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for BinaryCrossEntropyNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "binary_cross_entropy"
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
            let scale = g_out / n_t;

            let mut d_pred = vec![T::zero(); self.n];
            for i in 0..self.n {
                let p = self.probs[i];
                let t = self.targets[i];
                let one = T::one();
                // d/dp = -(t/p - (1-t)/(1-p)) / n
                // Use T::zero() - x idiom since unary - may not be in scope for T
                d_pred[i] = (T::zero() - (t / p) + (one - t) / (one - p)) * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d_pred, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Binary Cross-Entropy Loss.
/// pred: `[N]` probabilities (will be clamped internally), target: `[N]` float targets (0.0 or 1.0).
/// eps: numerical stability clamp (e.g., T::from_f64(1e-7)).
pub fn binary_cross_entropy<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    eps: T,
) -> Var<T, B> {
    let backend = B::default();
    let shape = pred.tensor.shape();
    let n = shape[0];

    // Host-side computation for forward + clamp
    let pred_cont;
    let pred_raw = if pred.tensor.is_contiguous() && pred.tensor.layout().offset() == 0 {
        &pred.tensor
    } else {
        pred_cont = pred.tensor.to_contiguous_on(&backend);
        &pred_cont
    };
    let target_cont;
    let target_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        target_cont = target.tensor.to_contiguous_on(&backend);
        &target_cont
    };

    let pred_host: std::borrow::Cow<[T]> = if let Some(s) = pred_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(pred_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let target_host: std::borrow::Cow<[T]> = if let Some(s) = target_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(target_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let one_minus_eps = T::one() - eps;
    let mut probs = vec![T::zero(); n];
    let mut targets_t = vec![T::zero(); n];
    let mut loss_val = T::zero();
    let n_t = T::from_f64(n as f64);

    for i in 0..n {
        let p_raw = pred_host[i];
        // Clamp to [eps, 1-eps]
        let p = if p_raw < eps {
            eps
        } else if p_raw > one_minus_eps {
            one_minus_eps
        } else {
            p_raw
        };
        let t = target_host[i];
        probs[i] = p;
        targets_t[i] = t;
        // -(t * log(p) + (1-t) * log(1-p)) using T::zero()-x for negation
        loss_val += T::zero() - (t * p.log_op() + (T::one() - t) * (T::one() - p).log_op());
    }
    loss_val = loss_val / n_t;

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(pred);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = BinaryCrossEntropyNode {
            output_grad,
            inputs: vec![pred.clone()],
            probs,
            targets: targets_t,
            n,
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
