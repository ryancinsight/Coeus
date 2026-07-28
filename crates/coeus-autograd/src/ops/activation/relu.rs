use super::unary_op;
use super::UnaryAutogradOp;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use crate::{mul, reshape, where_cond};
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST tag for ReLU autograd.
pub struct ReluOp;
impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ReluOp {
    const OP_NAME: &'static str = "relu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::relu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad)
            .expect("elementwise_unary");
        coeus_ops::mul(grad_out, &mask, backend)
    }
}

/// Tracked ReLU activation.
///
/// # Examples
///
/// `relu(x) = max(0, x)`; the gradient is 1 where `x > 0` and 0 otherwise.
/// For the scalar sum of `relu([2, -1])`, `dx = [1, 0]`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let x = Var::<f32, MoiraiBackend>::new(Tensor::from_slice([2], &[2.0, -1.0]), true);
/// let y = coeus_autograd::relu(&x);
/// assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-5);
/// assert!((y.tensor.as_slice()[1] - 0.0).abs() < 1e-5);
/// let loss = coeus_autograd::sum(&y);
/// loss.backward().expect("invariant: valid autograd fixture completes backward");
/// let grad = x.grad().unwrap();
/// assert!((grad.as_slice()[0] - 1.0).abs() < 1e-5); // x > 0
/// assert!((grad.as_slice()[1] - 0.0).abs() < 1e-5); // x < 0
/// ```
#[must_use]
#[inline]
pub fn relu<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ReluOp>(a)
}

/// Inline backward node for LeakyReLU.
struct LeakyReluNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    negative_slope: u64, // f64::to_bits
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for LeakyReluNode<T, B> {
    fn op_name(&self) -> &'static str {
        "leaky_relu"
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
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::LeakyReluGrad(self.negative_slope),
            )?;
            let mask = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &mask, &backend)?;
        }
        Ok(())
    }
}

/// Tracked Leaky ReLU activation.
#[must_use]
#[inline]
pub fn leaky_relu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    negative_slope: f64,
) -> Var<T, B> {
    let backend = B::default();
    let slope_bits = f64::to_bits(negative_slope);
    let out_tensor = coeus_ops::leaky_relu(&a.tensor, &backend, negative_slope);
    let requires_grad = crate::grad_mode::should_track_var(a);

    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let node = LeakyReluNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            negative_slope: slope_bits,
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

/// ZST tag for ELU autograd (alpha=1.0).
pub struct EluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for EluOp {
    const OP_NAME: &'static str = "elu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elu(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        // EluGrad takes the original input x and returns exp(x) or 1
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::EluGrad)
            .expect("elementwise_unary");
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked ELU activation.
#[must_use]
#[inline]
pub fn elu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, EluOp>(a)
}

/// SELU ELU parameter α.
const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
/// SELU scale parameter λ.
const SELU_SCALE: f64 = 1.050_700_987_355_480_5;

/// ZST tag for SELU autograd.
/// Backward: `scale` if `x > 0`, else `scale * alpha * exp(x)`.
pub struct SeluOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SeluOp {
    const OP_NAME: &'static str = "selu";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        let cond = coeus_ops::relu(x, backend);
        let scale = Tensor::full_on(x.shape(), T::from_f64(SELU_SCALE), backend);
        let alpha_scale = Tensor::full_on(x.shape(), T::from_f64(SELU_ALPHA * SELU_SCALE), backend);
        let pos = coeus_ops::mul(x, &scale, backend);
        let neg_base = coeus_ops::expm1(x, backend);
        let neg = coeus_ops::mul(&neg_base, &alpha_scale, backend);
        coeus_ops::where_cond(&cond, &pos, &neg, backend).expect("where_cond")
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let cond = coeus_ops::relu(x, backend);
        let scale = Tensor::full_on(x.shape(), T::from_f64(SELU_SCALE), backend);
        let alpha_scale = Tensor::full_on(x.shape(), T::from_f64(SELU_ALPHA * SELU_SCALE), backend);
        let neg = coeus_ops::mul(&coeus_ops::exp(x, backend), &alpha_scale, backend);
        let deriv = coeus_ops::where_cond(&cond, &scale, &neg, backend).expect("where_cond");
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked SELU activation.
#[must_use]
#[inline]
pub fn selu<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SeluOp>(a)
}

// ── PReLU (learnable weight) ──
//
// `y = x` where `x > 0`, else `weight · x` (PyTorch/Burn `PReLU` semantics —
// note the *kink at x = 0 lands on the negative branch*: PyTorch's `F.prelu`
// backward returns `weight`, not `1`, at x = 0). Expressed via the tracked
// `where_cond(cond, on_true, on_false)` — itself an existing composition of
// relu/mul/add with no custom `BackwardNode` of its own — selecting on
// `cond = relu(x)`, which is nonzero exactly where `x > 0`:
//   y = where_cond(relu(x), x, weight · x)
// `where_cond`'s own backward routes `grad_out` entirely to `on_true` where
// `cond ≠ 0` and entirely to `on_false` elsewhere (INCLUDING x = 0, since
// `relu(0) = 0`), so both PReLU gradients fall out correctly:
//   ∂y/∂x      = 1 where x > 0, else weight   (weight at the kink, matching
//                PyTorch — a plain `relu(x) - weight·relu(-x)` composition
//                would give 0 at the kink instead, since relu's own
//                subgradient there is 0, not a routed selection)
//   ∂y/∂weight = Σ_{x≤0} x   (broadcast-reduced to weight's shape; the x > 0
//                region contributes 0 since on_true doesn't depend on weight)
//
// `weight` has shape `[1]` (one shared slope) or `[C]` (per-channel,
// `num_parameters=C`). For rank > 2 inputs a `[C]` weight must broadcast
// against the CHANNEL axis (dim 1), not NumPy's default right-aligned trailing
// axis — e.g. `[N,C,H,W]` needs the weight reshaped to `[1,C,1,1]`.

/// Tracked PReLU with a learnable per-channel (or shared-scalar) weight.
///
/// `y = x` where `x > 0`, else `weight · x`. Gradient flows to both `x` (`1`
/// where `x > 0`, else `weight` — including at the kink `x = 0`, matching
/// PyTorch) and `weight` (`Σ_{x≤0} x`, broadcast-reduced).
///
/// # Panics
/// Panics (via the underlying broadcast) if `weight`'s channel count is
/// neither `1` nor `x`'s size along dim `1`.
#[must_use]
pub fn prelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    weight: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let input_rank = x.tensor.ndim();
    let channels = weight.tensor.shape()[0];
    let w = if channels > 1 && input_rank > 2 {
        let mut shape = vec![1usize; input_rank];
        shape[1] = channels;
        reshape(weight, shape)
    } else {
        weight.clone()
    };
    where_cond(&relu(x), x, &mul(&w, x))
}
