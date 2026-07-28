use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for KL divergence loss.
///
/// Computes `mean(target * (log(target) - input))` where `input` holds
/// log-probabilities and `target` holds probabilities. The gradient w.r.t.
/// `input` is `-target / N * grad_out` per element.
pub struct KlDivLossNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Target probabilities copied to host for backward.
    pub target_host: Vec<T>,
    /// Original input shape for backward gradient materialization.
    pub input_shape: Vec<usize>,
    /// Number of elements in the loss reduction.
    pub n: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for KlDivLossNode<T, B> {
    fn op_name(&self) -> &'static str {
        "kl_divergence"
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

            // d(loss)/d(input_i) = -target_i / N * grad_out
            let mut d_input = vec![T::zero(); self.n];
            for i in 0..self.n {
                d_input[i] = (T::zero() - self.target_host[i]) * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.input_shape.clone(), &d_input, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Tracked KL Divergence loss (PyTorch `F.kl_div` with `reduction='mean'`).
///
/// `input`: log-probabilities (log Q), `target`: probabilities (P).
/// Computes `mean(target * (log(target) - input))`.
/// Returns a scalar Var (shape `[1]`).
pub fn kl_divergence<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    let n = input.tensor.numel();
    let input_shape = input.tensor.shape().to_vec();
    assert_eq!(
        target.tensor.numel(),
        n,
        "input and target must have the same number of elements"
    );
    assert_eq!(
        target.tensor.shape(),
        input.tensor.shape(),
        "input and target must have the same shape"
    );

    let i_cont;
    let i_raw = if input.tensor.is_contiguous() && input.tensor.layout().offset() == 0 {
        &input.tensor
    } else {
        i_cont = input.tensor.to_contiguous_on(&backend);
        &i_cont
    };
    let t_cont;
    let t_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        t_cont = target.tensor.to_contiguous_on(&backend);
        &t_cont
    };

    let i_host: std::borrow::Cow<[T]> = if let Some(s) = i_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(i_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };
    let t_host: std::borrow::Cow<[T]> = if let Some(s) = t_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(t_raw.storage(), &mut v);
        std::borrow::Cow::Owned(v)
    };

    // loss = mean(target * (log(target) - input))
    // For target == 0, the term is 0 (0 * log(0) = 0 by convention).
    let mut loss_val = T::zero();
    let mut target_owned = vec![T::zero(); n];
    for i in 0..n {
        let p = t_host[i];
        target_owned[i] = p;
        let log_q = i_host[i];
        let term = if p == T::zero() {
            T::zero()
        } else {
            p * (p.log_op() - log_q)
        };
        loss_val += term;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = KlDivLossNode {
            output_grad,
            inputs: vec![input.clone()],
            target_host: target_owned,
            input_shape,
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
