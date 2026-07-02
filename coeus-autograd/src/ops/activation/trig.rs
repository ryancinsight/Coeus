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
