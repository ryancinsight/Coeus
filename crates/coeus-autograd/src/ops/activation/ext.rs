// ── Extended activation family (G-037 parity) ──
//
// Each function is implemented as a tracked autograd wrapper. Parameter-free
// variants (Hardsigmoid, Hardswish, Softsign) reuse the generic
// `unary_op<T, B, Op>(a)` ZST template. Parameterized variants
// (Hardtanh, Hardshrink, Softshrink, Threshold, Celu) share one generic
// monomorphized node/constructor path and only specialize the enum variant
// mapping for forward/backward dispatch.
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

/// Compile-time spec for parameterized unary autograd operations.
///
/// The spec is a zero-sized marker carried in the node only as a `PhantomData`,
/// so the `Send + Sync` supertraits are the ones a zero-sized marker already
/// satisfies; they are required because [`BackwardNode`] is itself `Send + Sync`.
trait ParameterizedUnarySpec: Send + Sync {
    const OP_NAME: &'static str;

    fn forward(bits: u64) -> coeus_ops::UnaryOp;
    fn backward(bits: u64) -> coeus_ops::UnaryOp;
}

/// Shared autograd node for parameterized unary operations.
struct ParameterizedUnaryNode<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    Spec: ParameterizedUnarySpec,
> {
    output_grad: Arc<GradBuffer<T, B>>,
    inputs: Vec<Var<T, B>>,
    input_tensor: Tensor<T, B>,
    bits: u64,
    _phantom: std::marker::PhantomData<Spec>,
}

impl<T, B, Spec> BackwardNode<T, B> for ParameterizedUnaryNode<T, B, Spec>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    Spec: ParameterizedUnarySpec,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        Spec::OP_NAME
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            let deriv = coeus_ops::elementwise_unary(
                &self.input_tensor,
                &backend,
                Spec::backward(self.bits),
            )?;
            let local = coeus_ops::mul(grad_out, &deriv, &backend)?;
            let lock = g.write();
            coeus_ops::add_assign(lock, &local, &backend)?;
        }
        Ok(())
    }
}

#[inline]
fn parameterized_unary_op<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    Spec: ParameterizedUnarySpec + 'static,
>(
    a: &Var<T, B>,
    bits: u64,
) -> Result<Var<T, B>, B::Error> {
    let backend = B::default();
    let out_tensor = coeus_ops::elementwise_unary(&a.tensor, &backend, Spec::forward(bits))?;
    let requires_grad = crate::grad_mode::should_track_var(a);
    Var::from_tracked_op(out_tensor, requires_grad, &backend, |output_grad| {
        ParameterizedUnaryNode::<T, B, Spec> {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            bits,
            _phantom: std::marker::PhantomData,
        }
    })
}

struct HardtanhSpec;

impl ParameterizedUnarySpec for HardtanhSpec {
    const OP_NAME: &'static str = "hardtanh";

    #[inline(always)]
    fn forward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::Hardtanh(bits)
    }

    #[inline(always)]
    fn backward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::HardtanhGrad(bits)
    }
}

/// Tracked Hardtanh: `y = clamp(x, min_val, max_val)`.
///
/// Gradient is the indicator `1_{min_val < x < max_val}`.
#[inline]
pub fn hardtanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    min_val: f64,
    max_val: f64,
) -> Result<Var<T, B>, B::Error> {
    let bits = pack_pairs(min_val, max_val);
    parameterized_unary_op::<T, B, HardtanhSpec>(a, bits)
}

// ── Hardsigmoid: y = clamp(x/6 + 0.5, 0, 1) ─────────────────────────────────

/// ZST tag for Hardsigmoid autograd (parameter-free).
pub struct HardsigmoidOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for HardsigmoidOp {
    const OP_NAME: &'static str = "hardsigmoid";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Hardsigmoid)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::HardsigmoidGrad)?;
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Hardsigmoid: `y = clamp(x/6 + 0.5, 0, 1)`.
///
/// Gradient is `1/6` in `(-3, 3)` and `0` outside.
#[inline]
pub fn hardsigmoid<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, HardsigmoidOp>(a)
}

// ── Hardswish: y = x · ReLU6(x+3) / 6 ───────────────────────────────────────

/// ZST tag for Hardswish autograd (parameter-free).
pub struct HardswishOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for HardswishOp {
    const OP_NAME: &'static str = "hardswish";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Hardswish)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::HardswishGrad)?;
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Hardswish: `y = x · clamp(x+3, 0, 6) / 6`.
///
/// Piecewise gradient: `0` for `x < -3`, `(2x+3)/6` for `-3 ≤ x ≤ 3`, `1`
/// for `x > 3`.
#[inline]
pub fn hardswish<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, HardswishOp>(a)
}

// ── Hardshrink: y = x if |x| > λ else 0 ────────────────────────────────────

struct HardshrinkSpec;

impl ParameterizedUnarySpec for HardshrinkSpec {
    const OP_NAME: &'static str = "hardshrink";

    #[inline(always)]
    fn forward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::Hardshrink(bits)
    }

    #[inline(always)]
    fn backward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::HardshrinkGrad(bits)
    }
}

/// Tracked Hardshrink: `y = (|x| > λ) ? x : 0`.
///
/// Gradient is `1` exactly where `|x| > λ`, `0` otherwise. The textbook
/// subgradient at `|x| = λ` is undefined; PyTorch's convention is `0` and we
/// match that here.
#[inline]
pub fn hardshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    lambda: f64,
) -> Result<Var<T, B>, B::Error> {
    let bits = lambda.to_bits();
    parameterized_unary_op::<T, B, HardshrinkSpec>(a, bits)
}

// ── Softshrink: y = sign(x) · max(|x| - λ, 0) ───────────────────────────────

struct SoftshrinkSpec;

impl ParameterizedUnarySpec for SoftshrinkSpec {
    const OP_NAME: &'static str = "softshrink";

    #[inline(always)]
    fn forward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::Softshrink(bits)
    }

    #[inline(always)]
    fn backward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::SoftshrinkGrad(bits)
    }
}

/// Tracked Softshrink: `y = sign(x) · max(|x| − λ, 0)`.
///
/// Gradient is `1` exactly where `|x| > λ`, `0` otherwise. Same subgradient
/// convention as Hardshrink.
#[inline]
pub fn softshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    lambda: f64,
) -> Result<Var<T, B>, B::Error> {
    let bits = lambda.to_bits();
    parameterized_unary_op::<T, B, SoftshrinkSpec>(a, bits)
}

// ── Softsign: y = x / (1 + |x|) ─────────────────────────────────────────────

/// ZST tag for Softsign autograd (parameter-free).
pub struct SoftsignOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SoftsignOp {
    const OP_NAME: &'static str = "softsign";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::Softsign)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        let deriv = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::SoftsignGrad)?;
        coeus_ops::mul(grad_out, &deriv, backend)
    }
}

/// Tracked Softsign: `y = x / (1 + |x|)`.
///
/// Gradient is `1 / (1 + |x|)^2`.
#[inline]
pub fn softsign<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, SoftsignOp>(a)
}

// ── Threshold: y = x if x > threshold else value ───────────────────────────

struct ThresholdSpec;

impl ParameterizedUnarySpec for ThresholdSpec {
    const OP_NAME: &'static str = "threshold";

    #[inline(always)]
    fn forward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::Threshold(bits)
    }

    #[inline(always)]
    fn backward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::ThresholdGrad(bits)
    }
}

/// Tracked Threshold: `y = (x > threshold) ? x : value`.
///
/// Gradient is `1` exactly when `x > threshold`, `0` otherwise. At the
/// kink `x = threshold` the replacement region dominates, so the
/// subgradient is `0` (PyTorch convention).
#[inline]
pub fn threshold<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    thresh: f64,
    value: f64,
) -> Result<Var<T, B>, B::Error> {
    let bits = pack_pairs(thresh, value);
    parameterized_unary_op::<T, B, ThresholdSpec>(a, bits)
}

// ── Celu: y = max(0,x) + min(0, α·(exp(x/α) − 1)) ───────────────────────────

struct CeluSpec;

impl ParameterizedUnarySpec for CeluSpec {
    const OP_NAME: &'static str = "celu";

    #[inline(always)]
    fn forward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::Celu(bits)
    }

    #[inline(always)]
    fn backward(bits: u64) -> coeus_ops::UnaryOp {
        coeus_ops::UnaryOp::CeluGrad(bits)
    }
}

/// Tracked Celu: `y = max(0, x) + min(0, α · (exp(x/α) − 1))`.
///
/// Gradient is `1` for `x ≥ 0`, else `exp(x/α)`. At the kink `x = 0`,
/// both pieces agree on derivative `1` (continuous-differentiable ELU).
#[inline]
pub fn celu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    alpha: f64,
) -> Result<Var<T, B>, B::Error> {
    let bits = alpha.to_bits();
    parameterized_unary_op::<T, B, CeluSpec>(a, bits)
}
