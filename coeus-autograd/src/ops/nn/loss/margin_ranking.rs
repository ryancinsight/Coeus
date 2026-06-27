use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for margin ranking loss.
///
/// Computes `mean(max(0, -target * (input1 - input2) + margin))`.
/// The gradient w.r.t. `input1` is `-target / N * grad_out` and w.r.t.
/// `input2` is `target / N * grad_out`, both only where the hinge is active.
pub struct MarginRankingLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation (input1, input2).
    pub inputs: Vec<Var<T, B>>,
    /// Target labels (+1 or -1) copied to host for backward.
    pub target_host: Vec<T>,
    /// Per-element hinge activation mask (1.0 if active, 0.0 if not).
    pub mask: Vec<T>,
    /// Number of elements in the loss reduction.
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for MarginRankingLossNode<T, B>
{
    fn op_name(&self) -> &'static str {
        "margin_ranking_loss"
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

        // d(loss)/d(input1_i) = -target_i * mask_i / N * grad_out
        // d(loss)/d(input2_i) =  target_i * mask_i / N * grad_out
        if let Some(Some(ref g1)) = input_grads.get(0) {
            let mut d1 = vec![T::zero(); self.n];
            for i in 0..self.n {
                d1[i] = (T::zero() - self.target_host[i]) * self.mask[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d1, &backend);
            let gl = g1.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
        if let Some(Some(ref g2)) = input_grads.get(1) {
            let mut d2 = vec![T::zero(); self.n];
            for i in 0..self.n {
                d2[i] = self.target_host[i] * self.mask[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on([self.n], &d2, &backend);
            let gl = g2.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Margin Ranking loss (PyTorch `F.margin_ranking_loss` with
/// `reduction='mean'`).
///
/// `input1`, `input2`: shape `[N]`, `target`: `&[T]` of +1 or -1, `margin`:
/// threshold. Computes `mean(max(0, -target * (input1 - input2) + margin))`.
/// Returns a scalar Var (shape `[1]`).
pub fn margin_ranking_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input1: &Var<T, B>,
    input2: &Var<T, B>,
    target: &[T],
    margin: T,
) -> Var<T, B> {
    let backend = B::default();
    let n = input1.tensor.numel();
    assert_eq!(
        input2.tensor.numel(),
        n,
        "input1 and input2 must have the same number of elements"
    );
    assert_eq!(target.len(), n, "target length must match input length");

    let i1_cont;
    let i1_raw = if input1.tensor.is_contiguous() && input1.tensor.layout().offset() == 0 {
        &input1.tensor
    } else {
        i1_cont = input1.tensor.to_contiguous_on(&backend);
        &i1_cont
    };
    let i2_cont;
    let i2_raw = if input2.tensor.is_contiguous() && input2.tensor.layout().offset() == 0 {
        &input2.tensor
    } else {
        i2_cont = input2.tensor.to_contiguous_on(&backend);
        &i2_cont
    };

    let i1_host: std::borrow::Cow<[T]> = if let Some(s) = i1_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(i1_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let i2_host: std::borrow::Cow<[T]> = if let Some(s) = i2_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(i2_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    let zero = T::zero();
    let mut loss_val = zero;
    let mut mask = vec![zero; n];
    for i in 0..n {
        let diff = i1_host[i] - i2_host[i];
        let raw = (T::zero() - target[i]) * diff + margin;
        let hinge = if raw > zero { raw } else { zero };
        if hinge > zero {
            mask[i] = T::one();
        }
        loss_val = loss_val + hinge;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(input1) || crate::grad_mode::should_track_var(input2);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = MarginRankingLossNode {
            output_grad,
            inputs: vec![input1.clone(), input2.clone()],
            target_host: target.to_vec(),
            mask,
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
