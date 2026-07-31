use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the row-wise p-norm pairwise distance.
pub struct PairwiseDistanceNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node (shape `[N]`).
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element `d(distance_i)/d(diff_ik)` factors, row-major `[N*D]`.
    pub grad_unit: Vec<T>,
    /// Number of rows `N`.
    pub rows: usize,
    /// Feature dimension `D`.
    pub feat: usize,
    /// Input shape `[N, D]` for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for PairwiseDistanceNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "pairwise_distance"
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
        let temp_grad;
        let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp_grad = grad_out.to_contiguous_on(&backend);
            &temp_grad
        };
        let mut g_rows = vec![T::zero(); self.rows];
        backend.copy_to_host(grad_cont.storage(), &mut g_rows);

        let want_x1 = matches!(input_grads.first(), Some(Some(_)));
        let want_x2 = matches!(input_grads.get(1), Some(Some(_)));
        if !want_x1 && !want_x2 {
            return Ok(());
        }

        let mut dx1 = vec![T::zero(); self.rows * self.feat];
        for i in 0..self.rows {
            let gi = g_rows[i];
            let base = i * self.feat;
            for k in 0..self.feat {
                dx1[base + k] = gi * self.grad_unit[base + k];
            }
        }

        if let Some(Some(ref g)) = input_grads.first() {
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &dx1, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut dx2 = dx1;
            for v in &mut dx2 {
                *v = T::zero() - *v;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &dx2, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Tracked row-wise p-norm pairwise distance (PyTorch `PairwiseDistance`):
/// for inputs `x1, x2` of shape `[N, D]`, returns a `[N]` vector with
/// `out_i = (sum_k |x1_ik - x2_ik + eps|^p)^(1/p)`. Matches PyTorch's
/// `torch.nn.functional.pairwise_distance` (`at::norm(x1 - x2 + eps, p)`)
/// exactly: `eps` is added to the difference, keeping the summed norm
/// strictly positive so the `s^(1/p - 1)` gradient factor stays finite.
pub fn pairwise_distance<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    p: T,
    eps: T,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        x1.tensor.shape(),
        x2.tensor.shape(),
        "pairwise_distance requires x1 and x2 to have identical shapes"
    );
    let shape_ref = x1.tensor.shape();
    assert_eq!(
        shape_ref.len(),
        2,
        "pairwise_distance expects 2D [N, D] inputs"
    );
    let rows = shape_ref[0];
    let feat = shape_ref[1];
    let n = rows * feat;
    let shape = x1.tensor.shape_cloned();

    let x1_cont;
    let x1_raw = if x1.tensor.is_contiguous() && x1.tensor.layout().offset() == 0 {
        &x1.tensor
    } else {
        x1_cont = x1.tensor.to_contiguous_on(&backend);
        &x1_cont
    };
    let x2_cont;
    let x2_raw = if x2.tensor.is_contiguous() && x2.tensor.layout().offset() == 0 {
        &x2.tensor
    } else {
        x2_cont = x2.tensor.to_contiguous_on(&backend);
        &x2_cont
    };

    let x1_host: std::borrow::Cow<[T]> = if let Some(s) = x1_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(x1_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let x2_host: std::borrow::Cow<[T]> = if let Some(s) = x2_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(x2_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let one = T::one();
    let inv_p = one / p;
    let p_minus_one = p - one;
    let mut out = vec![T::zero(); rows];
    let mut grad_unit = vec![T::zero(); n];
    for i in 0..rows {
        let base = i * feat;
        // PyTorch computes `at::norm(x1 - x2 + eps, p)`: `eps` is added to the
        // difference itself, not clamped onto the summed norm. Because every
        // shifted term contributes `eps^p > 0`, the norm is strictly positive,
        // which is exactly what keeps the `s^(1/p - 1)` gradient factor finite
        // (the reason torch adds `eps` here). It also resolves boundary cases —
        // e.g. an exactly-at-margin triplet — the same way torch does.
        let mut s = T::zero();
        for k in 0..feat {
            let diff = x1_host[base + k] - x2_host[base + k] + eps;
            s += <T as Float>::powf(<T as Float>::abs(diff), p);
        }
        let s_scaled = s.powf(inv_p);
        out[i] = s_scaled;
        let scale = s.powf(inv_p - one);
        for k in 0..feat {
            let diff = x1_host[base + k] - x2_host[base + k] + eps;
            let mag = <T as Float>::powf(<T as Float>::abs(diff), p_minus_one);
            let sign = if diff > T::zero() {
                one
            } else if diff < T::zero() {
                T::zero() - one
            } else {
                T::zero()
            };
            grad_unit[base + k] = scale * mag * sign;
        }
    }

    let out_tensor = Tensor::from_slice_on([rows], &out, &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(x1) || crate::grad_mode::should_track_var(x2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            [rows],
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = PairwiseDistanceNode {
            output_grad,
            inputs: vec![x1.clone(), x2.clone()],
            grad_unit,
            rows,
            feat,
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
