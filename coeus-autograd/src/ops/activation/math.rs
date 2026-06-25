use super::unary_op;
use super::UnaryAutogradOp;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, FloatOps, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── NegOp ──────────────────────────────────────────────────────────────────

/// ZST tag for negation autograd. Works for any `Scalar` (not just `Float`).
pub struct NegOp;
impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for NegOp {
    const OP_NAME: &'static str = "neg";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::neg(x, backend)
    }

    /// d/dx [−x] = −1, so grad_in = −grad_out.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::neg(grad_out, backend)
    }
}

/// Tracked element-wise negation.
#[must_use]
#[inline]
pub fn neg<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Var<T, B> {
    unary_op::<T, B, NegOp>(a)
}

// ── AbsOp ──────────────────────────────────────────────────────────────────

/// ZST tag for absolute-value autograd.
pub struct AbsOp;
impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AbsOp {
    const OP_NAME: &'static str = "abs";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::abs(x, backend)
    }

    /// d/dx |x| = sign(x).
    ///
    /// Computed as `abs(x) / x` where `x ≠ 0`.  At `x = 0` the result is 0
    /// (convention matches PyTorch/JAX).  Implemented via `abs(x)` then
    /// element-wise division using the backend's existing primitives — no new
    /// kernel required.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        // sign(x) = abs(x) / x; where x = 0 this produces NaN, masked to 0
        // by multiplying with ReluGrad(x) mask (0 at x≤0) combined with
        // ReluGrad(-x) (0 at x≥0). A cleaner and backend-portable approach:
        // use the stored forward output y = |x| and divide by x, then mul by grad_out.
        // Avoid division instability at x=0 by zeroing via the mask pattern:
        // sign = (x > 0) - (x < 0) encoded as abs(x)/x with zero at x=0 clamped.
        //
        // Implementation: sign(x) = neg(neg_mask) where:
        //   pos_mask = ReluGrad(x)      — 1 where x > 0, 0 elsewhere
        //   neg_mask = ReluGrad(-x_neg) — 1 where x < 0, 0 elsewhere
        // sign = pos_mask - neg_mask
        let pos_mask = coeus_ops::elementwise_unary(x, backend, coeus_ops::UnaryOp::ReluGrad);
        let x_neg = coeus_ops::neg(x, backend);
        let neg_mask = coeus_ops::elementwise_unary(&x_neg, backend, coeus_ops::UnaryOp::ReluGrad);
        let sign = coeus_ops::sub(&pos_mask, &neg_mask, backend);
        // Ignore the stored forward output `y` here — sign is cheaper directly from x.
        let _ = y;
        coeus_ops::mul(grad_out, &sign, backend)
    }
}

/// Tracked element-wise absolute value.
#[must_use]
#[inline]
pub fn abs<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Var<T, B> {
    unary_op::<T, B, AbsOp>(a)
}

// ── SqrtOp ─────────────────────────────────────────────────────────────────

/// ZST tag for square-root autograd.
pub struct SqrtOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SqrtOp {
    const OP_NAME: &'static str = "sqrt";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sqrt(x, backend)
    }

    /// d/dx √x = 1 / (2√x) = grad_out / (2·y) where y = √x.
    ///
    /// Uses the stored forward output `y` to avoid a redundant sqrt call.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let two = Tensor::full_on(y.shape(), T::from_f64(2.0), backend);
        let denom = coeus_ops::mul(y, &two, backend);
        coeus_ops::div(grad_out, &denom, backend)
    }
}

/// Tracked element-wise square root.
#[must_use]
#[inline]
pub fn sqrt<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SqrtOp>(a)
}

// ── PowNode ────────────────────────────────────────────────────────────────
//
// Parametric: stores `exp` as `u64` (bit-cast of f64) to stay `Sync`.
// Cannot use generic `UnaryNode<T, B, Op>` as `Op` would need to carry `exp`.
// Follows the same bespoke pattern as `LeakyReluNode`.

/// Autograd node for element-wise power: `y = x ^ exp`.
pub struct PowNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor snapshot for backward.
    pub input_tensor: Tensor<T, B>,
    /// Exponent stored as bit-cast `u64` for `Sync`-safe storage.
    pub exp_bits: u64,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for PowNode<T, B> {
    #[inline]
    fn op_name(&self) -> &'static str {
        "pow"
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
            let exp = f64::from_bits(self.exp_bits);
            let exp_t = T::from_f64(exp);
            let exp_m1 = exp - 1.0;
            // d/dx x^n = n · x^(n-1)
            // Compute x^(n-1) element-wise in native precision via ln/exp:
            //   x^(n-1) = exp((n-1) * ln(x))   for x > 0
            // For integer exponents or small n, this is the standard path.
            // A future optimization can use a dedicated pow kernel; for now
            // this composes existing backend primitives at zero cost.
            let ln_x = coeus_ops::log(&self.input_tensor, &backend);
            let scaled = {
                let scale = Tensor::full_on(ln_x.shape(), T::from_f64(exp_m1), &backend);
                coeus_ops::mul(&ln_x, &scale, &backend)
            };
            let x_pow_n_m1 = coeus_ops::exp(&scaled, &backend);
            let n_tensor = Tensor::full_on(x_pow_n_m1.shape(), exp_t, &backend);
            let local_grad = coeus_ops::mul(&n_tensor, &x_pow_n_m1, &backend);
            let grad_in = coeus_ops::mul(grad_out, &local_grad, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &grad_in, &backend);
        }
    }
}

/// Tracked element-wise power: `y = x ^ exp`.
///
/// `exp` is an `f64` scalar applied uniformly to all elements.
/// Backward: `d/dx x^n = n · x^(n−1)`.
///
/// # Precision
/// Forward and backward execute in the native precision of `T` without widening.
/// The exponent is converted once via `T::from_f64(exp)`.
#[must_use]
#[inline]
pub fn pow<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>, exp: f64) -> Var<T, B> {
    let backend = B::default();
    let exp_t = T::from_f64(exp);

    // Forward: x^n = exp(n * ln(x))
    let ln_x = coeus_ops::log(&a.tensor, &backend);
    let n_tensor = Tensor::full_on(ln_x.shape(), exp_t, &backend);
    let scaled = coeus_ops::mul(&n_tensor, &ln_x, &backend);
    let out_tensor = coeus_ops::exp(&scaled, &backend);

    let requires_grad = a.grad.is_some();
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
        let node = PowNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            exp_bits: f64::to_bits(exp),
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

// ── ClampNode ──────────────────────────────────────────────────────────────
//
// Gradient of clamp is 1 inside [min, max] and 0 outside.  This is the
// straight-through estimator for the STE variant; for exact grad it is the
// indicator function.  Here we implement the indicator (standard definition):
//
//   d/dx clamp(x, lo, hi) = 1_{lo ≤ x ≤ hi}
//
// which means: gradient passes through unchanged where the input is inside the
// clamp bounds, and is zeroed where x < lo or x > hi.
//
// Implementation via two ReluGrad applications on shifted inputs (zero-copy
// composition of existing backend primitives, no new kernel needed):
//   inside = ReluGrad(x - lo) * ReluGrad(hi - x)

/// Autograd node for element-wise clamp.
pub struct ClampNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor snapshot for backward mask computation.
    pub input_tensor: Tensor<T, B>,
    /// Lower bound of the clamp.
    pub min_val: T,
    /// Upper bound of the clamp.
    pub max_val: T,
}

impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for ClampNode<T, B>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "clamp"
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
            let shape = self.input_tensor.shape();
            // x - lo ≥ 0  ⟺  x ≥ lo
            let lo_t = Tensor::full_on(shape, self.min_val, &backend);
            let shifted_lo = coeus_ops::sub(&self.input_tensor, &lo_t, &backend);
            let mask_lo =
                coeus_ops::elementwise_unary(&shifted_lo, &backend, coeus_ops::UnaryOp::ReluGrad);
            // hi - x ≥ 0  ⟺  x ≤ hi
            let hi_t = Tensor::full_on(shape, self.max_val, &backend);
            let shifted_hi = coeus_ops::sub(&hi_t, &self.input_tensor, &backend);
            let mask_hi =
                coeus_ops::elementwise_unary(&shifted_hi, &backend, coeus_ops::UnaryOp::ReluGrad);
            // Inside mask = lo_mask AND hi_mask (product of indicators)
            let inside = coeus_ops::mul(&mask_lo, &mask_hi, &backend);
            let grad_in = coeus_ops::mul(grad_out, &inside, &backend);
            let lock = g.write();
            coeus_ops::add_assign(lock, &grad_in, &backend);
        }
    }
}

/// Tracked element-wise clamp: `y = clamp(x, min_val, max_val)`.
///
/// Gradient is the indicator `1_{min_val ≤ x ≤ max_val}` — passes through
/// where input is in range, zeroed at saturated positions.
#[must_use]
#[inline]
pub fn clamp<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    min_val: T,
    max_val: T,
) -> Var<T, B> {
    let backend = B::default();

    // Forward: clamp element-wise.
    // Implemented via existing primitives: clamp(x, lo, hi) = min(max(x, lo), hi)
    // = relu_shift approach using two relu operations on shifted values.
    // More precisely: clip from below then from above.
    let lo_t = Tensor::full_on(a.tensor.shape(), min_val, &backend);
    let hi_t = Tensor::full_on(a.tensor.shape(), max_val, &backend);
    // x_clamped_lo = relu(x - lo) + lo = max(x, lo)
    let shifted_lo = coeus_ops::sub(&a.tensor, &lo_t, &backend);
    let relu_lo = coeus_ops::elementwise_unary(&shifted_lo, &backend, coeus_ops::UnaryOp::Relu);
    let clamped_lo = coeus_ops::add(&relu_lo, &lo_t, &backend);
    // max(x, lo) clamped from above: min(clamped_lo, hi) = hi - relu(hi - clamped_lo)
    let shifted_hi = coeus_ops::sub(&hi_t, &clamped_lo, &backend);
    let relu_hi = coeus_ops::elementwise_unary(&shifted_hi, &backend, coeus_ops::UnaryOp::Relu);
    let out_tensor = coeus_ops::sub(&hi_t, &relu_hi, &backend);

    let requires_grad = a.grad.is_some();
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
        let node = ClampNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            min_val,
            max_val,
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

// ── RecipOp ────────────────────────────────────────────────────────────────

/// ZST tag for reciprocal autograd.
pub struct RecipOp;
impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B>
    for RecipOp
{
    const OP_NAME: &'static str = "recip";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::recip(x, backend)
    }

    /// d/dx [1/x] = −1/x² = −y² where y = 1/x.
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let y_sq = coeus_ops::mul(y, y, backend);
        let neg_y_sq = coeus_ops::neg(&y_sq, backend);
        coeus_ops::mul(grad_out, &neg_y_sq, backend)
    }
}

/// Tracked element-wise reciprocal.
#[must_use]
#[inline]
pub fn recip<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Var<T, B> {
    unary_op::<T, B, RecipOp>(a)
}

// ── Zero-gradient ops ───────────────────────────────────────────────────────
//
// Sign, floor, ceil, round, trunc are non-differentiable (or have zero
// gradient almost everywhere). Their backward returns a zero tensor.

macro_rules! impl_zero_grad_op {
    ($struct:ident, $func:ident, $op_name:expr, $forward_fn:ident) => {
        pub struct $struct;
        impl<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B>
            for $struct
        {
            const OP_NAME: &'static str = $op_name;

            #[inline(always)]
            fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
                coeus_ops::$forward_fn(x, backend)
            }

            /// Zero gradient — these ops are non-differentiable.
            #[inline(always)]
            fn backward(
                grad_out: &Tensor<T, B>,
                _x: &Tensor<T, B>,
                _y: &Tensor<T, B>,
                backend: &B,
            ) -> Tensor<T, B> {
                Tensor::zeros_on(grad_out.shape(), backend)
            }
        }

        /// Tracked element-wise `$op_name`.
        #[must_use]
        #[inline]
        pub fn $func<T: Scalar + FloatOps, B: coeus_ops::BackendOps<T> + Default>(
            a: &Var<T, B>,
        ) -> Var<T, B> {
            unary_op::<T, B, $struct>(a)
        }
    };
}

impl_zero_grad_op!(SignOp, sign, "sign", sign);
impl_zero_grad_op!(FloorOp, floor, "floor", floor);
impl_zero_grad_op!(CeilOp, ceil, "ceil", ceil);
impl_zero_grad_op!(RoundOp, round, "round", round);
impl_zero_grad_op!(TruncOp, trunc, "trunc", trunc);
