use super::unary_op;
use super::UnaryAutogradOp;
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatOps, Scalar};
use coeus_tensor::Tensor;
use std::ops::Neg;
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
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor snapshot for backward.
    pub input_tensor: Tensor<T, B>,
    /// Exponent stored as bit-cast `u64` for `Sync`-safe storage.
    pub exp_bits: u64,
}

impl<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B>
    for PowNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
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
        let Some(Some(ref g)) = input_grads.first() else {
            return;
        };

        let exp = f64::from_bits(self.exp_bits);
        let exp_t = T::from_f64(exp);
        let n = self.input_tensor.numel();
        if n == 0 {
            return;
        }

        // Host-side copy of upstream gradient.
        let temp_grad;
        let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp_grad = grad_out.to_contiguous_on(&backend);
            &temp_grad
        };
        let mut grad_host = vec![T::zero(); n];
        backend.copy_to_host(grad_cont.storage(), &mut grad_host);

        // Host-side copy of forward input.
        let input_contig =
            if self.input_tensor.is_contiguous() && self.input_tensor.layout().offset() == 0 {
                self.input_tensor.reshape([n])
            } else {
                self.input_tensor.to_contiguous_on(&backend).reshape([n])
            };
        let mut x_host = vec![T::zero(); n];
        backend.copy_to_host(input_contig.storage(), &mut x_host);

        // Decide whether `exp` is integer-valued in T.  When it is, the
        // PyTorch `Tensor.pow(scalar)` contract is sign-preserving integer power:
        //   (-x)^k = (-1)^k * x^k  (odd k negates, even k preserves).
        //   For k = 0 the result is 1 and d/dx = 0.
        // When `exp` is fractional, PyTorch follows IEEE: pow(x, p) = NaN
        // for x < 0; backward `n · x^(n-1)` mirrors that (NaN where x < 0).
        let exp_is_int = exp_t.is_integer();
        let exp_i = (exp as i64) as i32;

        let mut grad_in_host = vec![T::zero(); n];
        let one = T::one();

        if exp_is_int && exp_i == 0 {
            // x^0 = 1 → d/dx = 0; grad contribution is zero everywhere.
        } else if exp_is_int && exp_i > 0 {
            // d/dx x^k = k · x^(k-1).  Sign-preserving integer power:
            //   x^(k-1) = sign(x)^(k-1) · |x|^(k-1).
            // When k is odd, k-1 is even → sign^(k-1) = +1 (always non-negative).
            // When k is even, k-1 is odd  → sign^(k-1) = sign(x) (local grad may
            // change sign with x).
            let k = exp_i;
            let coef = exp_t;
            let k_m1 = (k - 1) as u32;
            let k_m1_odd = (k_m1 & 1) == 1;
            for i in 0..n {
                let x = x_host[i];
                if x == T::zero() {
                    // k = 1: d/dx x = 1; else x = 0 → grad = 0 (k > 1).
                    if k == 1 {
                        grad_in_host[i] = grad_host[i] * coef;
                    }
                    continue;
                }
                let abs_x = x.abs();
                let abs_pow_m1 = int_pow_positive(abs_x, k_m1);
                let local = if k_m1_odd {
                    // k-1 odd → sign(x) factor: x^(k-1) carries sign(x).
                    let sgn = if x < T::zero() { -one } else { one };
                    coef * sgn * abs_pow_m1
                } else {
                    // k-1 even → x^(k-1) is always non-negative.
                    coef * abs_pow_m1
                };
                grad_in_host[i] = grad_host[i] * local;
            }
        } else if exp_is_int && exp_i < 0 {
            // d/dx x^(-k) = -k · x^(-k-1) = -k / x^(k+1) (sign-preserving).
            // Denominator sign follows (k+1) parity on x.
            let k = (-exp_i) as u32;
            let neg_coef = -exp_t;
            let exp_total = k + 1;
            let exp_total_odd = (exp_total & 1) == 1;
            for i in 0..n {
                let x = x_host[i];
                if x == T::zero() {
                    // 1/0 = inf or NaN; PyTorch yields inf.  Leave as zero
                    // since `grad_host[i] * NaN` would taint the accumulator.
                    continue;
                }
                let abs_x = x.abs();
                let denom_abs = int_pow_positive(abs_x, exp_total);
                let denom = if exp_total_odd {
                    let sgn = if x < T::zero() { -one } else { one };
                    sgn * denom_abs
                } else {
                    denom_abs
                };
                let local = neg_coef / denom;
                grad_in_host[i] = grad_host[i] * local;
            }
        } else {
            // Fractional exponent path: compose with ln/exp so shape / device
            // multiplication is preserved via the backend dispatch.  For x < 0,
            // `ln(x)` propagates NaN (matches PyTorch IEEE).
            let exp_m1 = exp - 1.0;
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
            return;
        }

        let grad_t = Tensor::from_slice(self.input_tensor.shape().to_vec(), &grad_in_host);
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend);
    }
}

/// Repeated-squaring integer power in the native precision of `T`.
///
/// No widening, no `powf` fallback — straight `mul` in `T` so the integer-exponent
/// branch stays bit-identical to a hand-rolled `x^k = x·x·…·x`.  Exponent `0`
/// returns `T::one()`.  Used only on magnitude `|x| ≥ 0`; sign is applied by the
/// caller per the parity convention (`(-x)^k` is sign-preserving).
#[inline]
fn int_pow_positive<T: Float>(x: T, k: u32) -> T {
    let mut acc = T::one();
    let mut base = x;
    let mut e = k;
    while e > 0 {
        if (e & 1) == 1 {
            acc = acc * base;
        }
        base = base * base;
        e >>= 1;
    }
    acc
}

/// Tracked element-wise power: `y = x ^ exp`.
///
/// `exp` is an `f64` scalar applied uniformly to all elements.
/// Backward: `d/dx x^n = n · x^(n−1)`.
///
/// # Parity contract
///
/// Matches `torch.Tensor.pow(scalar)` semantics:
/// - When `exp` is integer-valued in `T` (i.e. `T::from_f64(exp).is_integer()`),
///   the forward is a sign-preserving integer power `(-x)^k = (-1)^k · x^k`,
///   with `x = 0` mapping to `1` when `k = 0`, to `0` when `k > 0`, and to
///   `+inf` when `k < 0` (delegated to PyTorch's IEEE-compliant convention).
/// - Otherwise forward composes `exp(n · ln(x))` (NaN for `x < 0`, IEEE).
///
/// # Precision
/// Forward and backward execute in the native precision of `T` without widening.
/// The exponent is converted once via `T::from_f64(exp)`; the integer-exponent
/// branch uses repeated multiplication in `T` (no `powf` fallback).
#[must_use]
#[inline]
pub fn pow<T: Float + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    exp: f64,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let exp_t = T::from_f64(exp);

    let n = a.tensor.numel();
    let out_tensor = if exp_t.is_integer() && n > 0 {
        // Host-fold sign-preserving integer power.
        let input_contig = if a.tensor.is_contiguous() && a.tensor.layout().offset() == 0 {
            a.tensor.reshape([n])
        } else {
            a.tensor.to_contiguous_on(&backend).reshape([n])
        };
        let mut x_host = vec![T::zero(); n];
        backend.copy_to_host(input_contig.storage(), &mut x_host);

        let exp_i = (exp as i64) as i32;
        let one = T::one();
        let mut out_host = vec![T::zero(); n];
        if exp_i == 0 {
            for v in &mut out_host {
                *v = one;
            }
        } else if exp_i > 0 {
            let k = exp_i as u32;
            for i in 0..n {
                let x = x_host[i];
                if x == T::zero() {
                    // x^0 = 1 already handled above; x^k for k > 0 is 0.
                    out_host[i] = T::zero();
                    continue;
                }
                let abs_x = x.abs();
                let abs_pow = int_pow_positive(abs_x, k);
                out_host[i] = if (k & 1) == 1 {
                    let sgn = if x < T::zero() { -one } else { one };
                    sgn * abs_pow
                } else {
                    abs_pow
                };
            }
        } else {
            // exp_i < 0: x^(-k) = 1 / x^k with sign preservation.
            let k = (-exp_i) as u32;
            for i in 0..n {
                let x = x_host[i];
                if x == T::zero() {
                    // 1/0 → +inf per PyTorch IEEE; emit +inf to mirror that
                    // (avoids NaN from sign*0 division).
                    out_host[i] = T::INFINITY;
                    continue;
                }
                let abs_x = x.abs();
                let denom_abs = int_pow_positive(abs_x, k);
                let denom = if (k & 1) == 1 {
                    let sgn = if x < T::zero() { -one } else { one };
                    sgn * denom_abs
                } else {
                    denom_abs
                };
                out_host[i] = one / denom;
            }
        }
        Tensor::from_slice(a.tensor.shape().to_vec(), &out_host)
    } else {
        // Fractional exponent path: x^n = exp(n · ln(x)).  NaN for x ≤ 0
        // (IEEE) is preserved by `ln(x)` propagation through the backend.
        let ln_x = coeus_ops::log(&a.tensor, &backend);
        let n_tensor = Tensor::full_on(ln_x.shape(), exp_t, &backend);
        let scaled = coeus_ops::mul(&n_tensor, &ln_x, &backend);
        coeus_ops::exp(&scaled, &backend)
    };

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
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
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
            let one_scalar = Tensor::full_on(shape, T::one(), &backend);

            // 1_{lo <= x} = 1 - 1_{x < lo} = 1 - ReluGrad(lo - x).
            // At the kink x = lo: (lo - x) = 0, ReluGrad(0) = 0 (strict),
            // so the indicator evaluates to 1 — matching PyTorch's
            // `aten::clamp_backward_kernel` which is `[min, max]` inclusive.
            let lo_t = Tensor::full_on(shape, self.min_val, &backend);
            let lo_minus_x = coeus_ops::sub(&lo_t, &self.input_tensor, &backend);
            let mask_lt_lo =
                coeus_ops::elementwise_unary(&lo_minus_x, &backend, coeus_ops::UnaryOp::ReluGrad);
            let mask_lo_ge = coeus_ops::sub(&one_scalar, &mask_lt_lo, &backend);

            // 1_{x <= hi} = 1 - 1_{x > hi} = 1 - ReluGrad(x - hi).
            // At the kink x = hi: (x - hi) = 0, ReluGrad(0) = 0 (strict),
            // so the upper indicator evaluates to 1 — same inclusive
            // convention as the lower bound.
            let hi_t = Tensor::full_on(shape, self.max_val, &backend);
            let x_minus_hi = coeus_ops::sub(&self.input_tensor, &hi_t, &backend);
            let mask_gt_hi =
                coeus_ops::elementwise_unary(&x_minus_hi, &backend, coeus_ops::UnaryOp::ReluGrad);
            let mask_hi_le = coeus_ops::sub(&one_scalar, &mask_gt_hi, &backend);

            // Inside mask = mask_lo_ge AND mask_hi_le (product of indicators)
            let inside = coeus_ops::mul(&mask_lo_ge, &mask_hi_le, &backend);
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
        #[doc = concat!("Zero-gradient unary op marker for `", stringify!($struct), "`.")]
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
