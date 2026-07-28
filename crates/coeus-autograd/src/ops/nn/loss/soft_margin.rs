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

/// Autograd node for the soft-margin (logistic) loss.
pub struct SoftMarginNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Per-element `-target * sigmoid(-target*input)`, the `d/d_input` factor.
    pub d_input: Vec<T>,
    /// Per-element `-input * sigmoid(-target*input)`, the `d/d_target` factor.
    pub d_target: Vec<T>,
    /// Number of elements in the mean reduction.
    pub n: usize,
    /// Original tensor shape for gradient reconstruction.
    pub shape: coeus_core::Shape,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SoftMarginNode<T, B> {
    fn op_name(&self) -> &'static str {
        "soft_margin"
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

        if let Some(Some(ref g)) = input_grads.first() {
            let mut d = vec![T::zero(); self.n];
            for (i, grad) in d.iter_mut().enumerate() {
                *grad = self.d_input[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        if let Some(Some(ref g)) = input_grads.get(1) {
            let mut d = vec![T::zero(); self.n];
            for (i, grad) in d.iter_mut().enumerate() {
                *grad = self.d_target[i] * scale;
            }
            let grad_tensor = Tensor::from_slice_on(self.shape.clone(), &d, &backend);
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_tensor, &backend)?;
        }

        Ok(())
    }
}

/// Tracked soft-margin (logistic) loss, the `reduction="mean"` form of PyTorch
/// `SoftMarginLoss`: `mean_i log(1 + exp(-target_i * input_i))`, with `target`
/// in `{-1, +1}`. `input` and `target` must share shape.
///
/// Forward uses the stable softplus identity
/// `log(1+exp(-m)) = max(-m, 0) + log(1 + exp(-|m|))` with `m = target*input`.
/// Differentiable w.r.t. both inputs:
/// `d/d_input = -target * sigmoid(-m) / n`, `d/d_target = -input * sigmoid(-m) / n`.
pub fn soft_margin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let backend = B::default();
    assert_eq!(
        input.tensor.shape(),
        target.tensor.shape(),
        "soft_margin requires input and target to have identical shapes"
    );
    let n = input.tensor.numel();
    assert!(n > 0, "soft_margin requires at least one element");
    let shape = input.tensor.shape_cloned();

    let x_cont;
    let x_raw = if input.tensor.is_contiguous() && input.tensor.layout().offset() == 0 {
        &input.tensor
    } else {
        x_cont = input.tensor.to_contiguous_on(&backend);
        &x_cont
    };
    let t_cont;
    let t_raw = if target.tensor.is_contiguous() && target.tensor.layout().offset() == 0 {
        &target.tensor
    } else {
        t_cont = target.tensor.to_contiguous_on(&backend);
        &t_cont
    };

    let x_host: std::borrow::Cow<[T]> = if let Some(s) = x_raw.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&s[..n])
    } else {
        let mut v = vec![T::zero(); n];
        backend.copy_to_host(x_raw.storage(), &mut v);
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
    let mut d_input = vec![T::zero(); n];
    let mut d_target = vec![T::zero(); n];
    let mut loss_val = T::zero();
    for i in 0..n {
        let x = x_host[i];
        let y = t_host[i];
        let m = y * x;
        // softplus(-m) = max(-m, 0) + log(1 + exp(-|m|)).
        let neg_m = T::zero() - m;
        let max_part = if neg_m > T::zero() { neg_m } else { T::zero() };
        loss_val =
            loss_val + max_part + (one + (T::zero() - <T as Float>::abs(m)).exp_op()).log_op();
        // sigmoid(-m); grads: d/dx = -y*sig, d/dy = -x*sig.
        let sig = stable_sigmoid(neg_m);
        d_input[i] = (T::zero() - y) * sig;
        d_target[i] = (T::zero() - x) * sig;
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
        let node = SoftMarginNode {
            output_grad,
            inputs: vec![input.clone(), target.clone()],
            d_input,
            d_target,
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
