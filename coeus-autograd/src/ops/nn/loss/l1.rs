use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for L1 (mean absolute error) loss.
pub struct L1LossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise differences `pred[i] - target[i]`, stored for backward.
    pub diffs: Vec<T>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for L1LossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "l1_loss"
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
        // d/d_pred mean|pred - target| = sign(pred - target) / n.
        // Subgradient at the kink is 0, matching torch.sign(0) == 0.
        let neg_one = T::zero() - T::one();
        let mut d_pred = vec![T::zero(); self.n];
        for (i, grad) in d_pred.iter_mut().enumerate() {
            let diff = self.diffs[i];
            let sign = if diff > T::zero() {
                T::one()
            } else if diff < T::zero() {
                neg_one
            } else {
                T::zero()
            };
            *grad = sign * scale;
        }

        if let Some(Some(ref g)) = input_grads.first() {
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_pred, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }

        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut d_target = d_pred;
            for grad in &mut d_target {
                *grad = T::zero() - *grad;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_target, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked L1 (mean absolute error) loss: `mean_i |pred[i] - target[i]|`.
/// pred and target must have identical shape. The mean reduction covers every
/// element, not only the leading dimension.
pub fn l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        pred.tensor.shape(),
        target.tensor.shape(),
        "l1_loss requires pred and target to have identical shapes"
    );
    let n = pred.tensor.numel();
    assert!(n > 0, "l1_loss requires at least one element");
    let shape = pred.tensor.shape_cloned();

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

    let mut diffs = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let diff = p_host[i] - t_host[i];
        diffs[i] = diff;
        let abs_diff = if diff < T::zero() {
            T::zero() - diff
        } else {
            diff
        };
        loss_val += abs_diff;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(pred) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = L1LossNode {
            output_grad,
            inputs: vec![pred.clone(), target.clone()],
            diffs,
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
