use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for Huber loss (PyTorch `F.huber_loss(reduction='mean')`).
///
/// The forward uses the **classical** Huber definition — which differs from
/// `smooth_l1_loss` by omitting the `1/δ` factor in the quadratic region
/// and rescaling the linear region by `δ`:
///
///   forward quadratic (`|z| < δ`): `0.5 * z²`
///   forward linear    (`|z| ≥ δ`): `δ * |z| - 0.5 * δ²`
///   backward quadratic: `z`
///   backward linear:   `sign(z) * δ`
pub struct HuberLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise differences `pred[i] - target[i]`, stored for backward.
    pub diffs: Vec<T>,
    /// Delta threshold separating quadratic from linear regions.
    pub delta: T,
    /// Number of elements in the loss reduction.
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
                // PyTorch's huber_loss gradient: `z` in the quadratic region
                // and `sign(z) * delta` in the linear region. (Note: this
                // differs from smooth_l1_loss, whose quadratic grad is
                // `z / beta` — huber_loss uses the classical definition.)
                let abs_diff = if diff < T::zero() {
                    T::zero() - diff
                } else {
                    diff
                };
                let gradient = if abs_diff <= delta {
                    diff
                } else {
                    // Preserve the sign of `diff` (i.e. `sign(diff) * delta`).
                    let sign = if diff < T::zero() {
                        T::zero() - T::one()
                    } else {
                        T::one()
                    };
                    sign * delta
                };
                d_pred[i] = gradient * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d_pred, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Huber loss (PyTorch `F.huber_loss(reduction='mean', delta=...)`).
///
/// Forward per element `z = pred - target`:
///   - quadratic branch (`|z| < delta`): `0.5 * z * z`
///   - linear branch   (`|z| >= delta`): `delta * |z| - 0.5 * delta * delta`
///
/// Backward per element:
///   - quadratic: `z`
///   - linear:   `sign(z) * delta`
///
/// This matches the classical Huber definition; PyTorch's
/// `smooth_l1_loss` is the `0.5·z²/β`-form alternative.
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
        // Classical Huber: `0.5*z²` for |z|<δ, `δ*|z| - 0.5*δ²` otherwise.
        // (PyTorch's huber_loss differs from smooth_l1_loss by omitting the
        // `1/δ` factor in the quadratic region:
        //   smooth_l1: 0.5*z²/β for |z|<β, |z|-0.5*β otherwise
        //   huber:     0.5*z²   for |z|<δ, δ*|z|-0.5*δ² otherwise)
        let elem = if abs_diff <= delta {
            half * diff * diff
        } else {
            delta * abs_diff - half * delta * delta
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
