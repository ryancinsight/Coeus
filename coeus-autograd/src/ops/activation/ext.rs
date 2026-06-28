// ── Extended activation family (G-037 parity) ──
//
// Each function is implemented as a tracked autograd wrapper. Parameter-free
// variants (Hardsigmoid, Hardswish, Softsign) reuse the generic
// `unary_op<T, B, Op>(a)` ZST template. Parameterized variants
// (Hardtanh, Hardshrink, Softshrink, Threshold, Celu) follow the manual
// `LeakyReluNode` pattern from `relu.rs`: a per-call struct holding the
// packed-scalar parameters, plus a hand-written constructor that attaches
// the creator node.
//
// Subgradient contract at kink points mirrors PyTorch's convention:
//   - Hardtanh at x = min_val or x = max_val: gradient passes through as 1.0.
//   - Hardsigmoid at x = -3 or x = 3: gradient is exactly 1/6.
//   - Hardswish at x = -3 or x = 3: gradient is (2x+3)/6 = -0.5 / 1.5.
//   - Hardshrink / Softshrink at |x| = λ: gradient is 0 (post-kink convention).
//   - Threshold at x = threshold: gradient is 0 (replacement region).
//   - Celu at x = 0: gradient is 1 (continuously differentiable).
//
// All parameterized scalar values pass through `f64::to_bits` packed into a
// `u64` field on `CpuUnaryOp` (see `coeus-core::CpuUnaryOp` decode conventions).

use super::unary_op;
use super::UnaryAutogradOp;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── Bit-packing helpers ─────────────────────────────────────────────────────

/// Pack two scalar parameter values into a single `u64` as little-endian `f32`
/// bit patterns.
///
/// Layout (LSB->MSB): `bits[0..32] = (low as f32).to_bits()`,
/// `bits[32..64] = (high as f32).to_bits()`. `CpuUnaryOp` decodes each half as
/// `f32` and then converts to the active scalar type.
#[inline]
pub fn pack_pairs(low: f64, high: f64) -> u64 {
    let low = (low as f32).to_bits() as u64;
    let high = ((high as f32).to_bits() as u64) << 32;
    low | high
}

// ── Hardtanh: y = clamp(x, min_val, max_val) ────────────────────────────────

/// Manual autograd node for Hardtanh (parameterized min/max).
struct HardtanhNode<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for HardtanhNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "hardtanh"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::HardtanhGrad(self.bits),
            );
            let local = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend);
        }
    }
}

/// Tracked Hardtanh: `y = clamp(x, min_val, max_val)`.
///
/// Gradient is the indicator `1_{min_val < x < max_val}`.
#[must_use]
#[inline]
pub fn hardtanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    min_val: f64,
    max_val: f64,
) -> Var<T, B> {
    let backend = B::default();
    let bits = pack_pairs(min_val, max_val);
    let out_tensor =
        coeus_ops::elementwise_unary(&a.tensor, &backend, coeus_ops::UnaryOp::Hardtanh(bits));
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
        let node = HardtanhNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
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

// ── Hardsigmoid: y = clamp(x/6 + 0.5, 0, 1) ─────────────────────────────────

/// ZST tag for Hardsigmoid autograd (parameter-free).
pub struct HardsigmoidOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for HardsigmoidOp {
    const OP_NAME: &'static str = "hardsigmoid";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Hardsigmoid)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::HardsigmoidGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Hardsigmoid: `y = clamp(x/6 + 0.5, 0, 1)`.
///
/// Gradient is `1/6` in `(-3, 3)` and `0` outside.
#[must_use]
#[inline]
pub fn hardsigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, HardsigmoidOp>(a)
}

// ── Hardswish: y = x · ReLU6(x+3) / 6 ───────────────────────────────────────

/// ZST tag for Hardswish autograd (parameter-free).
pub struct HardswishOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for HardswishOp {
    const OP_NAME: &'static str = "hardswish";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Hardswish)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::HardswishGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Hardswish: `y = x · clamp(x+3, 0, 6) / 6`.
///
/// Piecewise gradient: `0` for `x < -3`, `(2x+3)/6` for `-3 ≤ x ≤ 3`, `1`
/// for `x > 3`.
#[must_use]
#[inline]
pub fn hardswish<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, HardswishOp>(a)
}

// ── Hardshrink: y = x if |x| > λ else 0 ────────────────────────────────────

/// Manual autograd node for Hardshrink (parameterized λ).
struct HardshrinkNode<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for HardshrinkNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "hardshrink"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::HardshrinkGrad(self.bits),
            );
            let local = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend);
        }
    }
}

/// Tracked Hardshrink: `y = (|x| > λ) ? x : 0`.
///
/// Gradient is `1` exactly where `|x| > λ`, `0` otherwise. The textbook
/// subgradient at `|x| = λ` is undefined; PyTorch's convention is `0` and we
/// match that here.
#[must_use]
#[inline]
pub fn hardshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    lambda: f64,
) -> Var<T, B> {
    let backend = B::default();
    let bits = lambda.to_bits();
    let out_tensor =
        coeus_ops::elementwise_unary(&a.tensor, &backend, coeus_ops::UnaryOp::Hardshrink(bits));
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
        let node = HardshrinkNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
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

// ── Softshrink: y = sign(x) · max(|x| - λ, 0) ───────────────────────────────

/// Manual autograd node for Softshrink (parameterized λ).
struct SoftshrinkNode<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for SoftshrinkNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "softshrink"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::SoftshrinkGrad(self.bits),
            );
            let local = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend);
        }
    }
}

/// Tracked Softshrink: `y = sign(x) · max(|x| − λ, 0)`.
///
/// Gradient is `1` exactly where `|x| > λ`, `0` otherwise. Same subgradient
/// convention as Hardshrink.
#[must_use]
#[inline]
pub fn softshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    lambda: f64,
) -> Var<T, B> {
    let backend = B::default();
    let bits = lambda.to_bits();
    let out_tensor =
        coeus_ops::elementwise_unary(&a.tensor, &backend, coeus_ops::UnaryOp::Softshrink(bits));
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
        let node = SoftshrinkNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
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

// ── Softsign: y = x / (1 + |x|) ─────────────────────────────────────────────

/// ZST tag for Softsign autograd (parameter-free).
pub struct SoftsignOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SoftsignOp {
    const OP_NAME: &'static str = "softsign";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Softsign)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SoftsignGrad);
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Softsign: `y = x / (1 + |x|)`.
///
/// Gradient is `1 / (1 + |x|)^2`.
#[must_use]
#[inline]
pub fn softsign<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SoftsignOp>(a)
}

// ── Threshold: y = x if x > threshold else value ───────────────────────────

/// Manual autograd node for Threshold (parameterized threshold + value).
struct ThresholdNode<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for ThresholdNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "threshold"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::ThresholdGrad(self.bits),
            );
            let local = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend);
        }
    }
}

/// Tracked Threshold: `y = (x > threshold) ? x : value`.
///
/// Gradient is `1` exactly when `x > threshold`, `0` otherwise. At the
/// kink `x = threshold` the replacement region dominates, so the
/// subgradient is `0` (PyTorch convention).
#[must_use]
#[inline]
pub fn threshold<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    thresh: f64,
    value: f64,
) -> Var<T, B> {
    let backend = B::default();
    let bits = pack_pairs(thresh, value);
    let out_tensor =
        coeus_ops::elementwise_unary(&a.tensor, &backend, coeus_ops::UnaryOp::Threshold(bits));
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
        let node = ThresholdNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
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

// ── Celu: y = max(0,x) + min(0, α·(exp(x/α) − 1)) ───────────────────────────

/// Manual autograd node for Celu (parameterized α).
struct CeluNode<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CeluNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "celu"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                coeus_ops::UnaryOp::CeluGrad(self.bits),
            );
            let local = coeus_ops::mul(grad_out, &deriv, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend);
        }
    }
}

/// Tracked Celu: `y = max(0, x) + min(0, α · (exp(x/α) − 1))`.
///
/// Gradient is `1` for `x ≥ 0`, else `exp(x/α)`. At the kink `x = 0`,
/// both pieces agree on derivative `1` (continuous-differentiable ELU).
#[must_use]
#[inline]
pub fn celu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    alpha: f64,
) -> Var<T, B> {
    let backend = B::default();
    let bits = alpha.to_bits();
    let out_tensor =
        coeus_ops::elementwise_unary(&a.tensor, &backend, coeus_ops::UnaryOp::Celu(bits));
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
        let node = CeluNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
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
