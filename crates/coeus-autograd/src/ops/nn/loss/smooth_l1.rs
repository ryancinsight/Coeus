//! Smooth L1 (Huber-β) loss autograd node.
//!
//! PyTorch contract (`SmoothL1Loss(reduction="mean", beta=1.0)`):
//!   loss(n) = 0.5 * (z_n)^2 / beta           if |z_n| < beta
//!           = |z_n| - 0.5 * beta              otherwise
//!
//! where `z = pred - target`. The gradient is the standard subgradient
//! matching PyTorch's `F.smooth_l1_loss`:
//!   d/dz = z / beta                          if |z| < beta
//!        = sign(z)                           otherwise
//!
//! At the kink (`|z| == beta`) Coeus takes the **right limit** to match
//! PyTorch's numerically stable reduction (the L1 piece, gradient `sign(z)`).
//! This avoids an implementation-defined subgradient at the boundary and
//! is empirically verified via parity tests against `torch.nn.functional.smooth_l1_loss`.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for Smooth L1 loss (mean reduction).
pub struct SmoothL1LossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the scalar output.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Element-wise `z = pred - target`, stored for the gradient routing.
    pub diffs: Vec<T>,
    /// Threshold β (encoded as `T::from_f64(self.beta_f64)` at construction).
    pub beta: T,
    /// Per-element gradient factor `d/d_pred = g_out / n * (z/β if |z|<β else sign(z))`.
    pub grad_pred: Vec<T>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for SmoothL1LossNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "smooth_l1_loss"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let grad_cont;
        let grad_src = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            grad_cont = grad_out.to_contiguous_on(&backend);
            &grad_cont
        };
        let mut host_grad = [T::zero()];
        backend.copy_to_host(grad_src.storage(), &mut host_grad);
        let g_out = host_grad[0];
        let n_t = T::from_f64(self.n as f64);
        let scale = g_out / n_t;

        // d/d_pred: scale * (z / beta) if |z| < beta, else scale * sign(z).
        let neg_one = T::zero() - T::one();
        let inv_beta = T::one() / self.beta;
        let mut d_pred = vec![T::zero(); self.n];
        for (i, grad) in d_pred.iter_mut().enumerate() {
            let z = self.diffs[i];
            let abs_z = if z < T::zero() { T::zero() - z } else { z };
            *grad = if abs_z < self.beta {
                z * inv_beta * scale
            } else if z > T::zero() {
                scale
            } else if z < T::zero() {
                neg_one * scale
            } else {
                // |z| == beta: take the right limit (L1 piece).
                // sign(0) -> 0 to match PyTorch's reduce-at-zero behavior.
                T::zero()
            };
        }

        if let Some(Some(ref g)) = input_grads.first() {
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_pred, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut d_target = d_pred;
            for grad in &mut d_target {
                *grad = T::zero() - *grad;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_target, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Tracked Smooth L1 loss: `mean_i loss_smooth(pred[i] - target[i], beta)`.
/// `pred` and `target` must share shape. Mirrors PyTorch
/// `SmoothL1Loss(reduction="mean", beta=float)`.
pub fn smooth_l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    beta: T,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        pred.tensor.shape(),
        target.tensor.shape(),
        "smooth_l1_loss requires pred and target to have identical shapes"
    );
    assert!(
        beta > T::zero(),
        "smooth_l1_loss requires beta > 0; got beta = 0 (would divide by zero)"
    );
    let n = pred.tensor.numel();
    assert!(n > 0, "smooth_l1_loss requires at least one element");
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

    let inv_beta = T::one() / beta;
    let half = T::from_f64(0.5);
    let mut diffs = vec![T::zero(); n];
    let mut grad_pred = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let z = p_host[i] - t_host[i];
        diffs[i] = z;
        let abs_z = if z < T::zero() { T::zero() - z } else { z };
        let elem = if abs_z < beta {
            half * z * z * inv_beta
        } else {
            abs_z - half * beta
        };
        loss_val += elem;
        // Pre-stage the per-element d/d_pred factor (sans the outer
        // (g_out / n) scale; the node applies the scale at backward time).
        let unit_grad = if abs_z < beta {
            z * inv_beta
        } else if z > T::zero() {
            T::one()
        } else if z < T::zero() {
            T::zero() - T::one()
        } else {
            // At |z| == beta: take the right limit (L1 piece); sign(0) = 0.
            T::zero()
        };
        grad_pred[i] = unit_grad;
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
    let creator = grad.as_ref().cloned().map(|output_grad| {
        let node = SmoothL1LossNode {
            output_grad,
            inputs: vec![pred.clone(), target.clone()],
            diffs,
            beta,
            grad_pred,
            n,
            shape,
        };
        Arc::new(node) as Arc<dyn BackwardNode<T, B>>
    });
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
