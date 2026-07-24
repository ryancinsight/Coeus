use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar, Storage};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for the Poisson negative-log-likelihood loss (log-input form).
pub struct PoissonNllNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element `exp(input) - target`, the `d/d_input` factor before scaling.
    pub exp_minus_target: Vec<T>,
    /// Per-element inputs, the `-d/d_target` factor before scaling.
    pub input_vals: Vec<T>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PoissonNllNode<T, B> {
    fn op_name(&self) -> &'static str {
        "poisson_nll"
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

        // d/d_input = (exp(input) - target) / n.
        if let Some(Some(ref g)) = input_grads.first() {
            let mut d_input = vec![T::zero(); self.n];
            for (i, grad) in d_input.iter_mut().enumerate() {
                *grad = self.exp_minus_target[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_input, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }

        // d/d_target = -input / n.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut d_target = vec![T::zero(); self.n];
            for (i, grad) in d_target.iter_mut().enumerate() {
                *grad = (T::zero() - self.input_vals[i]) * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d_target, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend);
        }
    }
}

/// Tracked Poisson negative-log-likelihood loss in the log-input regime
/// (PyTorch `PoissonNLLLoss(log_input=True, full=False, reduction="mean")`):
/// `loss = mean_i(exp(input_i) - target_i * input_i)`.
///
/// `input` holds the log-rate `log(λ)`, `target` the observed counts; both
/// share shape. The Stirling `full` correction term is not included (matching
/// the PyTorch default).
pub fn poisson_nll<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        input.tensor.shape(),
        target.tensor.shape(),
        "poisson_nll requires input and target to have identical shapes"
    );
    let n = input.tensor.numel();
    assert!(n > 0, "poisson_nll requires at least one element");
    let shape = input.tensor.shape_cloned();

    let z_cont;
    let z_raw = if input.tensor.is_contiguous() && input.tensor.layout().offset() == 0 {
        &input.tensor
    } else {
        z_cont = input.tensor.to_contiguous_on(&backend);
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

    let mut exp_minus_target = vec![T::zero(); n];
    let mut input_vals = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let z = z_host[i];
        let y = t_host[i];
        let ez = z.exp_op();
        input_vals[i] = z;
        exp_minus_target[i] = ez - y;
        loss_val += ez - y * z;
    }
    loss_val = loss_val / T::from_f64(n as f64);

    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);
    let requires_grad =
        crate::grad_mode::should_track_var(input) || crate::grad_mode::should_track_var(target);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on([1], &backend))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = PoissonNllNode {
            output_grad,
            inputs: vec![input.clone(), target.clone()],
            exp_minus_target,
            input_vals,
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
