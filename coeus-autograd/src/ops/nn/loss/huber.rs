use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct HuberLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise differences `pred[i] - target[i]`, stored for backward.
    pub diffs: Vec<T>,
    pub delta: T,
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for HuberLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "huber_loss"
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
            let delta = self.delta;

            let mut d_pred = vec![T::zero(); self.n];
            for i in 0..self.n {
                let diff = self.diffs[i];
                // Huber grad: diff/delta clamped to [-1, 1]
                let raw = diff / delta;
                // Use T::zero() - T::one() for -1.0 since unary - not guaranteed on T
                let neg_one = T::zero() - T::one();
                let clamped = if raw > T::one() {
                    T::one()
                } else if raw < neg_one {
                    neg_one
                } else {
                    raw
                };
                d_pred[i] = clamped * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d_pred, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Huber (Smooth L1) Loss.
/// pred: `[N]`, target: `[N]`, delta: threshold.
pub fn huber_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = pred.tensor.shape()[0];

    let p_cont;
    let p_raw = if pred.tensor.is_contiguous() && pred.tensor.layout().offset() == 0 {
        &pred.tensor
    } else {
        p_cont = pred.tensor.to_contiguous_on(&backend);
        &p_cont
    };
    let t_cont;
    let t_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        t_cont = target.tensor.to_contiguous_on(&backend);
        &t_cont
    };

    let p_host: std::borrow::Cow<[T]> = if let Some(s) = p_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(p_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let t_host: std::borrow::Cow<[T]> = if let Some(s) = t_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(t_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let half = T::from_f64(0.5);
    let mut diffs = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let diff = p_host[i] - t_host[i];
        diffs[i] = diff;
        // abs_diff using T::zero() - diff for negation
        let abs_diff = if diff < T::zero() {
            T::zero() - diff
        } else {
            diff
        };
        let elem = if abs_diff <= delta {
            half * diff * diff / delta
        } else {
            abs_diff - half * delta
        };
        loss_val = loss_val + elem;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(pred);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = HuberLossNode {
            output_grad,
            inputs: vec![pred.clone()],
            diffs,
            delta,
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
