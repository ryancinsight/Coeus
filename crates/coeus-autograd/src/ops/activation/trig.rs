use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

// ── Exponential / logarithm ────────────────────────────────────────────────

unary_autograd!(ExpOp, "exp", exp, |g, _x, y, b| coeus_ops::mul(g, y, b));

unary_autograd!(LogOp, "log", log, |g, x, _y, b| coeus_ops::div(g, x, b));

// ── Erf ─────────────────────────────────────────────────────────────────────
//
// Backward by the fundamental theorem of calculus: `d/dx erf(x) = (2/√π)·e^(−x²)`,
// `d/dx erfc(x) = −(2/√π)·e^(−x²)`. Both compose the existing
// mul/exp/neg/scalar primitives (no dedicated gradient kernel needed).

unary_autograd!(ErfOp, "erf", erf, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let neg_x_sq = coeus_ops::neg(&x_sq, b);
    let gauss = coeus_ops::exp(&neg_x_sq, b);
    let two_over_sqrt_pi = Tensor::full_on(
        gauss.shape(),
        T::from_f64(core::f64::consts::FRAC_2_SQRT_PI),
        b,
    );
    let scaled = coeus_ops::mul(&gauss, &two_over_sqrt_pi, b);
    coeus_ops::mul(g, &scaled, b)
});

unary_autograd!(ErfcOp, "erfc", erfc, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let neg_x_sq = coeus_ops::neg(&x_sq, b);
    let gauss = coeus_ops::exp(&neg_x_sq, b);
    let neg_two_over_sqrt_pi = Tensor::full_on(
        gauss.shape(),
        T::from_f64(-core::f64::consts::FRAC_2_SQRT_PI),
        b,
    );
    let scaled = coeus_ops::mul(&gauss, &neg_two_over_sqrt_pi, b);
    coeus_ops::mul(g, &scaled, b)
});

// ── Sin / cos / tan ────────────────────────────────────────────────────────

// `d/dx sin(x) = cos(x)`. Recomputes `cos` from the input `x` rather than the
// stored output `y = sin(x)` (the `sqrt(1 - y²)` identity only holds for small x).
unary_autograd!(SinOp, "sin", sin, |g, x, _y, b| {
    let cos_x = coeus_ops::cos(x, b);
    coeus_ops::mul(g, &cos_x, b)
});

unary_autograd!(CosOp, "cos", cos, |g, x, _y, b| {
    let sin_x = coeus_ops::sin(x, b);
    let neg_sin_x = coeus_ops::neg(&sin_x, b);
    coeus_ops::mul(g, &neg_sin_x, b)
});

// `d/dx tan(x) = sec²(x) = 1 / cos²(x)`.
unary_autograd!(TanOp, "tan", tan, |g, x, _y, b| {
    let cos_x = coeus_ops::cos(x, b);
    let cos_sq = coeus_ops::mul(&cos_x, &cos_x, b);
    let inv_cos_sq = coeus_ops::recip(&cos_sq, b);
    coeus_ops::mul(g, &inv_cos_sq, b)
});

// ── Inverse trigonometric ──────────────────────────────────────────────────

// `d/dx asin(x) = 1/√(1 − x²)`.
unary_autograd!(AsinOp, "asin", asin, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let one_minus_xsq = coeus_ops::sub(&one, &x_sq, b);
    let sqrt_val = coeus_ops::sqrt(&one_minus_xsq, b);
    let inv_sqrt = coeus_ops::recip(&sqrt_val, b);
    coeus_ops::mul(g, &inv_sqrt, b)
});

// `d/dx acos(x) = −1/√(1 − x²)`.
unary_autograd!(AcosOp, "acos", acos, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let one_minus_xsq = coeus_ops::sub(&one, &x_sq, b);
    let sqrt_val = coeus_ops::sqrt(&one_minus_xsq, b);
    let inv_sqrt = coeus_ops::recip(&sqrt_val, b);
    let neg_inv_sqrt = coeus_ops::neg(&inv_sqrt, b);
    coeus_ops::mul(g, &neg_inv_sqrt, b)
});

// `d/dx atan(x) = 1/(1 + x²)`.
unary_autograd!(AtanOp, "atan", atan, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let one_plus_xsq = coeus_ops::add(&one, &x_sq, b);
    let inv = coeus_ops::recip(&one_plus_xsq, b);
    coeus_ops::mul(g, &inv, b)
});

// ── Hyperbolic ─────────────────────────────────────────────────────────────

unary_autograd!(SinhOp, "sinh", sinh, |g, x, _y, b| {
    let cosh_x = coeus_ops::cosh(x, b);
    coeus_ops::mul(g, &cosh_x, b)
});

unary_autograd!(CoshOp, "cosh", cosh, |g, x, _y, b| {
    let sinh_x = coeus_ops::sinh(x, b);
    coeus_ops::mul(g, &sinh_x, b)
});

// `d/dx atanh(x) = 1/(1 − x²)`.
unary_autograd!(AtanhOp, "atanh", atanh, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let one_minus_xsq = coeus_ops::sub(&one, &x_sq, b);
    let inv = coeus_ops::recip(&one_minus_xsq, b);
    coeus_ops::mul(g, &inv, b)
});

// `d/dx asinh(x) = 1/√(x² + 1)`.
unary_autograd!(AsinhOp, "asinh", asinh, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let xsq_plus_one = coeus_ops::add(&x_sq, &one, b);
    let sqrt_val = coeus_ops::sqrt(&xsq_plus_one, b);
    let inv = coeus_ops::recip(&sqrt_val, b);
    coeus_ops::mul(g, &inv, b)
});

// `d/dx acosh(x) = 1/√(x² − 1)`.
unary_autograd!(AcoshOp, "acosh", acosh, |g, x, _y, b| {
    let x_sq = coeus_ops::mul(x, x, b);
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let xsq_minus_one = coeus_ops::sub(&x_sq, &one, b);
    let sqrt_val = coeus_ops::sqrt(&xsq_minus_one, b);
    let inv = coeus_ops::recip(&sqrt_val, b);
    coeus_ops::mul(g, &inv, b)
});

// ── Base-2 / base-10 log and exp-near-one ──────────────────────────────────

// `d/dx log2(x) = 1/(x·ln 2)`.
unary_autograd!(Log2Op, "log2", log2, |g, x, _y, b| {
    let ln2 = Tensor::full_on(x.shape(), T::from_f64(core::f64::consts::LN_2), b);
    let x_ln2 = coeus_ops::mul(x, &ln2, b);
    let inv = coeus_ops::recip(&x_ln2, b);
    coeus_ops::mul(g, &inv, b)
});

// `d/dx log10(x) = 1/(x·ln 10)`.
unary_autograd!(Log10Op, "log10", log10, |g, x, _y, b| {
    let ln10 = Tensor::full_on(x.shape(), T::from_f64(core::f64::consts::LN_10), b);
    let x_ln10 = coeus_ops::mul(x, &ln10, b);
    let inv = coeus_ops::recip(&x_ln10, b);
    coeus_ops::mul(g, &inv, b)
});

// `d/dx exp2(x) = 2ˣ·ln 2`, using the stored output `y = 2ˣ`.
unary_autograd!(Exp2Op, "exp2", exp2, |g, _x, y, b| {
    let ln2 = Tensor::full_on(y.shape(), T::from_f64(core::f64::consts::LN_2), b);
    let y_ln2 = coeus_ops::mul(y, &ln2, b);
    coeus_ops::mul(g, &y_ln2, b)
});

// `d/dx expm1(x) = exp(x)`.
unary_autograd!(Expm1Op, "expm1", expm1, |g, x, _y, b| {
    let exp_x = coeus_ops::exp(x, b);
    coeus_ops::mul(g, &exp_x, b)
});

// `d/dx log1p(x) = 1/(1 + x)`.
unary_autograd!(Log1pOp, "log1p", log1p, |g, x, _y, b| {
    let one = Tensor::full_on(x.shape(), T::one(), b);
    let one_plus_x = coeus_ops::add(&one, x, b);
    let inv = coeus_ops::recip(&one_plus_x, b);
    coeus_ops::mul(g, &inv, b)
});

// ── Forward-only ───────────────────────────────────────────────────────────

/// Forward-only natural logarithm of the absolute gamma function.
///
/// This is not exported as a differentiable autograd node because the
/// derivative is `digamma(x)`, which is not available in the current
/// datatype-law surface.
#[must_use]
#[inline]
pub fn lgamma_forward<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Tensor<T, B> {
    let backend = B::default();
    coeus_ops::lgamma(&a.tensor, &backend)
}
