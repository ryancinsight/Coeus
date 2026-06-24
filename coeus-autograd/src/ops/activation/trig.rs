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
