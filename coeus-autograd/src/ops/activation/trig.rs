use super::unary_op;
use super::UnaryAutogradOp;
use crate::var::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

/// ZST tag for Exponential autograd.
pub struct ExpOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ExpOp {
    const OP_NAME: &'static str = "exp";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::exp(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::mul(grad_out, y, backend)
    }
}

/// ZST tag for Natural Logarithm autograd.
pub struct LogOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for LogOp {
    const OP_NAME: &'static str = "log";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        coeus_ops::div(grad_out, x, backend)
    }
}

// ── Erf ─────────────────────────────────────────────────────────────────────

/// ZST tag for Gauss-error-function autograd.
///
/// Forward: `erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt`.  Backward: by the fundamental
/// theorem of calculus, `d/dx erf(x) = (2/√π) · e^(−x²)`, so
/// `grad_in = grad_out * (2/√π) * exp(-x²)` — composed from the existing
/// mul/exp/neg/scalar primitives (no dedicated gradient kernel needed).
pub struct ErfOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ErfOp {
    const OP_NAME: &'static str = "erf";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::erf(x, backend)
    }

    /// d/dx erf(x) = (2/√π)·e^(−x²).
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let neg_x_sq = coeus_ops::neg(&x_sq, backend);
        let gauss = coeus_ops::exp(&neg_x_sq, backend);
        let two_over_sqrt_pi = Tensor::full_on(
            gauss.shape(),
            T::from_f64(core::f64::consts::FRAC_2_SQRT_PI),
            backend,
        );
        let scaled = coeus_ops::mul(&gauss, &two_over_sqrt_pi, backend);
        coeus_ops::mul(grad_out, &scaled, backend)
    }
}

/// ZST tag for Complementary Error Function autograd.
///
/// Forward: `erfc(x) = 1 − erf(x)`.  Backward: `d/dx erfc(x) = −(2/√π)·e^(−x²)`.
pub struct ErfcOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for ErfcOp {
    const OP_NAME: &'static str = "erfc";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::erfc(x, backend)
    }

    /// d/dx erfc(x) = -(2/√π)·e^(−x²)  (negation of erf gradient).
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let neg_x_sq = coeus_ops::neg(&x_sq, backend);
        let gauss = coeus_ops::exp(&neg_x_sq, backend);
        let neg_two_over_sqrt_pi = Tensor::full_on(
            gauss.shape(),
            T::from_f64(-core::f64::consts::FRAC_2_SQRT_PI),
            backend,
        );
        let scaled = coeus_ops::mul(&gauss, &neg_two_over_sqrt_pi, backend);
        coeus_ops::mul(grad_out, &scaled, backend)
    }
}

// ── Sin ─────────────────────────────────────────────────────────────────────

/// ZST tag for Sine autograd.
///
/// Forward: `sin(x)`.  Backward: `d/dx sin(x) = cos(x)`, so
/// `grad_in = grad_out * cos(x)`.  Uses the *input* `x` (not the stored
/// forward output `y = sin(x)`) to recover `cos(x)` without a separate
/// branch: `cos(x) = sqrt(1 - y²)` only holds for small `x`, so we
/// recompute `cos` from `x` directly.
pub struct SinOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SinOp {
    const OP_NAME: &'static str = "sin";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sin(x, backend)
    }

    /// d/dx sin(x) = cos(x).
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let cos_x = coeus_ops::cos(x, backend);
        coeus_ops::mul(grad_out, &cos_x, backend)
    }
}

// ── Cos ─────────────────────────────────────────────────────────────────────

/// ZST tag for Cosine autograd.
///
/// Forward: `cos(x)`.  Backward: `d/dx cos(x) = -sin(x)`, so
/// `grad_in = grad_out * (-sin(x))`.
pub struct CosOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for CosOp {
    const OP_NAME: &'static str = "cos";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::cos(x, backend)
    }

    /// d/dx cos(x) = -sin(x).
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let sin_x = coeus_ops::sin(x, backend);
        let neg_sin_x = coeus_ops::neg(&sin_x, backend);
        coeus_ops::mul(grad_out, &neg_sin_x, backend)
    }
}

/// ZST tag for Tangent autograd.
/// Forward: tan(x). Backward: d/dx tan(x) = 1/cos²(x) = sec²(x).
/// grad_in = grad_out * (1 / (cos(x) * cos(x)))
pub struct TanOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for TanOp {
    const OP_NAME: &'static str = "tan";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::tan(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let cos_x = coeus_ops::cos(x, backend);
        let cos_sq = coeus_ops::mul(&cos_x, &cos_x, backend);
        let inv_cos_sq = coeus_ops::recip(&cos_sq, backend);
        coeus_ops::mul(grad_out, &inv_cos_sq, backend)
    }
}

/// ZST tag for Arc-Sine autograd.
/// Forward: asin(x). Backward: d/dx asin(x) = 1/sqrt(1 - x²).
pub struct AsinOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AsinOp {
    const OP_NAME: &'static str = "asin";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::asin(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let one_minus_xsq = coeus_ops::sub(&one, &x_sq, backend);
        let sqrt_val = coeus_ops::sqrt(&one_minus_xsq, backend);
        let inv_sqrt = coeus_ops::recip(&sqrt_val, backend);
        coeus_ops::mul(grad_out, &inv_sqrt, backend)
    }
}

/// ZST tag for Arc-Cosine autograd.
/// Forward: acos(x). Backward: d/dx acos(x) = -1/sqrt(1 - x²).
pub struct AcosOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AcosOp {
    const OP_NAME: &'static str = "acos";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::acos(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let one_minus_xsq = coeus_ops::sub(&one, &x_sq, backend);
        let sqrt_val = coeus_ops::sqrt(&one_minus_xsq, backend);
        let inv_sqrt = coeus_ops::recip(&sqrt_val, backend);
        let neg_inv_sqrt = coeus_ops::neg(&inv_sqrt, backend);
        coeus_ops::mul(grad_out, &neg_inv_sqrt, backend)
    }
}

/// ZST tag for Arc-Tangent autograd.
/// Forward: atan(x). Backward: d/dx atan(x) = 1/(1 + x²).
pub struct AtanOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AtanOp {
    const OP_NAME: &'static str = "atan";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::atan(x, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let one_plus_xsq = coeus_ops::add(&one, &x_sq, backend);
        let inv = coeus_ops::recip(&one_plus_xsq, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for Hyperbolic Sine autograd.
/// Backward: d/dx sinh(x) = cosh(x).
pub struct SinhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for SinhOp {
    const OP_NAME: &'static str = "sinh";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::sinh(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let cosh_x = coeus_ops::cosh(x, backend);
        coeus_ops::mul(grad_out, &cosh_x, backend)
    }
}

/// ZST tag for Hyperbolic Cosine autograd.
/// Backward: d/dx cosh(x) = sinh(x).
pub struct CoshOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for CoshOp {
    const OP_NAME: &'static str = "cosh";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::cosh(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let sinh_x = coeus_ops::sinh(x, backend);
        coeus_ops::mul(grad_out, &sinh_x, backend)
    }
}

/// ZST tag for base-2 log autograd.
/// Backward: d/dx log2(x) = 1/(x * ln(2)).
pub struct Log2Op;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Log2Op {
    const OP_NAME: &'static str = "log2";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log2(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let ln2 = Tensor::full_on(x.shape(), T::from_f64(core::f64::consts::LN_2), backend);
        let x_ln2 = coeus_ops::mul(x, &ln2, backend);
        let inv = coeus_ops::recip(&x_ln2, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for base-10 log autograd.
/// Backward: d/dx log10(x) = 1/(x * ln(10)).
pub struct Log10Op;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Log10Op {
    const OP_NAME: &'static str = "log10";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log10(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let ln10 = Tensor::full_on(x.shape(), T::from_f64(core::f64::consts::LN_10), backend);
        let x_ln10 = coeus_ops::mul(x, &ln10, backend);
        let inv = coeus_ops::recip(&x_ln10, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for inverse hyperbolic tangent autograd.
/// Backward: d/dx atanh(x) = 1/(1 - x²).
pub struct AtanhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AtanhOp {
    const OP_NAME: &'static str = "atanh";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::atanh(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let one_minus_xsq = coeus_ops::sub(&one, &x_sq, backend);
        let inv = coeus_ops::recip(&one_minus_xsq, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for inverse hyperbolic sine autograd.
/// Backward: d/dx asinh(x) = 1/sqrt(x² + 1).
pub struct AsinhOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AsinhOp {
    const OP_NAME: &'static str = "asinh";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::asinh(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let xsq_plus_one = coeus_ops::add(&x_sq, &one, backend);
        let sqrt_val = coeus_ops::sqrt(&xsq_plus_one, backend);
        let inv = coeus_ops::recip(&sqrt_val, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for inverse hyperbolic cosine autograd.
/// Backward: d/dx acosh(x) = 1/sqrt(x² - 1).
pub struct AcoshOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for AcoshOp {
    const OP_NAME: &'static str = "acosh";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::acosh(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let x_sq = coeus_ops::mul(x, x, backend);
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let xsq_minus_one = coeus_ops::sub(&x_sq, &one, backend);
        let sqrt_val = coeus_ops::sqrt(&xsq_minus_one, backend);
        let inv = coeus_ops::recip(&sqrt_val, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// ZST tag for expm1 autograd.
/// Backward: d/dx expm1(x) = exp(x).
pub struct Expm1Op;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Expm1Op {
    const OP_NAME: &'static str = "expm1";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::expm1(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let exp_x = coeus_ops::exp(x, backend);
        coeus_ops::mul(grad_out, &exp_x, backend)
    }
}

/// ZST tag for log1p autograd.
/// Backward: d/dx log1p(x) = 1/(1 + x).
pub struct Log1pOp;
impl<T: Float, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Log1pOp {
    const OP_NAME: &'static str = "log1p";
    #[inline(always)]
    fn forward(x: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
        coeus_ops::log1p(x, backend)
    }
    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        backend: &B,
    ) -> Tensor<T, B> {
        let one = Tensor::full_on(x.shape(), T::one(), backend);
        let one_plus_x = coeus_ops::add(&one, x, backend);
        let inv = coeus_ops::recip(&one_plus_x, backend);
        coeus_ops::mul(grad_out, &inv, backend)
    }
}

/// Tracked Exponential function.
#[must_use]
#[inline]
pub fn exp<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ExpOp>(a)
}

/// Tracked Natural Logarithm.
#[must_use]
#[inline]
pub fn log<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, LogOp>(a)
}

/// Tracked Gauss error function.
///
/// Backward: `d/dx erf(x) = (2/√π)·e^(−x²)`.
#[must_use]
#[inline]
pub fn erf<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ErfOp>(a)
}

/// Tracked complementary error function.
///
/// Backward: `d/dx erfc(x) = -(2/√π)·e^(−x²)`.
#[must_use]
#[inline]
pub fn erfc<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, ErfcOp>(a)
}

/// Tracked element-wise sine.
///
/// Backward: `d/dx sin(x) = cos(x)`.
#[must_use]
#[inline]
pub fn sin<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SinOp>(a)
}

/// Tracked element-wise cosine.
///
/// Backward: `d/dx cos(x) = -sin(x)`.
#[must_use]
#[inline]
pub fn cos<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, CosOp>(a)
}

/// Tracked tan.
#[must_use]
#[inline]
pub fn tan<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, TanOp>(a)
}

/// Tracked asin.
#[must_use]
#[inline]
pub fn asin<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AsinOp>(a)
}

/// Tracked acos.
#[must_use]
#[inline]
pub fn acos<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AcosOp>(a)
}

/// Tracked atan.
#[must_use]
#[inline]
pub fn atan<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AtanOp>(a)
}

/// Tracked element-wise hyperbolic sine.
#[must_use]
#[inline]
pub fn sinh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, SinhOp>(a)
}

/// Tracked element-wise hyperbolic cosine.
#[must_use]
#[inline]
pub fn cosh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, CoshOp>(a)
}

/// Tracked element-wise base-2 logarithm.
#[must_use]
#[inline]
pub fn log2<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, Log2Op>(a)
}

/// Tracked element-wise base-10 logarithm.
#[must_use]
#[inline]
pub fn log10<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, Log10Op>(a)
}

/// Tracked element-wise inverse hyperbolic tangent.
#[must_use]
#[inline]
pub fn atanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AtanhOp>(a)
}

/// Tracked element-wise inverse hyperbolic sine.
#[must_use]
#[inline]
pub fn asinh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AsinhOp>(a)
}

/// Tracked element-wise inverse hyperbolic cosine.
#[must_use]
#[inline]
pub fn acosh<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, AcoshOp>(a)
}

/// Tracked element-wise exp(x) - 1.
#[must_use]
#[inline]
pub fn expm1<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, Expm1Op>(a)
}

/// Tracked element-wise ln(1 + x).
#[must_use]
#[inline]
pub fn log1p<T: Float, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, Log1pOp>(a)
}
