use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Numerically stable sigmoid `1 / (1 + exp(-z))` without intermediate overflow.
#[inline]
fn stable_sigmoid<T: Float>(z: T) -> T {
    let one = T::one();
    if z >= T::zero() {
        one / (one + (T::zero() - z).exp_op())
    } else {
        let ez = z.exp_op();
        ez / (one + ez)
    }
}

/// Autograd node for binary cross-entropy computed from logits.
pub struct BceWithLogitsNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element `sigmoid(logit) - target`, the `d/d_logit` factor before scaling.
    pub sig_minus_target: Vec<T>,
    /// Per-element logits, the `-d/d_target` factor before scaling.
    pub logits: Vec<T>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for BceWithLogitsNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "bce_with_logits"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
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

        // d/d_logit = (sigmoid(z) - y) / n.
        if let Some(Some(ref g)) = input_grads.first() {
            let mut d_logit = vec![T::zero(); self.n];
            for (i, grad) in d_logit.iter_mut().enumerate() {
                *grad = self.sig_minus_target[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_logit, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }

        // d/d_target = -z / n.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut d_target = vec![T::zero(); self.n];
            for (i, grad) in d_target.iter_mut().enumerate() {
                *grad = (T::zero() - self.logits[i]) * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_target, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked binary cross-entropy with logits: the numerically stable composition
/// of a sigmoid and binary cross-entropy. `logits` and `target` must share shape.
///
/// Per element with `z = logit`, `y = target`:
/// `loss = max(z, 0) - z*y + log(1 + exp(-|z|))`, averaged over all elements.
/// This is the `reduction="mean"` form of PyTorch `BCEWithLogitsLoss`.
pub fn bce_with_logits<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        logits.tensor.shape(),
        target.tensor.shape(),
        "bce_with_logits requires logits and target to have identical shapes"
    );
    let n = logits.tensor.numel();
    assert!(n > 0, "bce_with_logits requires at least one element");
    let shape = logits.tensor.shape_cloned();

    let z_cont;
    let z_raw = if logits.tensor.is_contiguous() && logits.tensor.layout().offset() == 0 {
        &logits.tensor
    } else {
        z_cont = logits.tensor.to_contiguous_on(&backend);
        &z_cont
    };
    let t_cont;
    let t_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        t_cont = target.tensor.to_contiguous_on(&backend);
        &t_cont
    };

    let z_host: std::borrow::Cow<[T]> = if let Some(s) = z_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(z_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let t_host: std::borrow::Cow<[T]> = if let Some(s) = t_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(t_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let one = T::one();
    let mut sig_minus_target = vec![T::zero(); n];
    let mut logits_v = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let z = z_host[i];
        let y = t_host[i];
        logits_v[i] = z;
        // Stable: max(z,0) - z*y + log(1 + exp(-|z|)).
        let max_z0 = if z > T::zero() { z } else { T::zero() };
        let elem = max_z0 - z * y + (one + (T::zero() - <T as Float>::abs(z)).exp_op()).log_op();
        loss_val += elem;
        sig_minus_target[i] = stable_sigmoid(z) - y;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(logits) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = BceWithLogitsNode {
            output_grad,
            inputs: vec![logits.clone(), target.clone()],
            sig_minus_target,
            logits: logits_v,
            n,
            shape,
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
