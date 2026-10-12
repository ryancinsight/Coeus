use crate::tensor::dispatch::{apply_binary, apply_unary};
use crate::tensor::PyTensor;
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pyfunction]
pub fn exp(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::exp)
}

#[pyfunction]
pub fn log(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::log)
}

/// Gauss error function (`torch.erf`); differentiable, d/dx = 2/√π·e^(−x²).
#[pyfunction]
pub fn erf(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::erf)
}

/// Complementary error function (`torch.special.erfc`); differentiable.
#[pyfunction]
pub fn erfc(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::erfc)
}

/// Natural logarithm of the absolute gamma function (`torch.special.gammaln`).
///
/// The forward path is available for non-grad tensors. Gradients require
/// `digamma`, which is not available in the current provider surface.
#[pyfunction]
pub fn gammaln(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    if input.inner.grad.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "gammaln backward requires digamma support",
        ));
    }
    let tensor = py.allow_threads(|| coeus_autograd::lgamma_forward(&input.inner));
    Ok(PyTensor::from_var(Var::new(tensor, false)))
}

/// Alias matching `torch.lgamma`.
#[pyfunction]
pub fn lgamma(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    gammaln(input, py)
}

#[pyfunction]
pub fn tan(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::tan)
}

#[pyfunction]
pub fn asin(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::asin)
}

#[pyfunction]
pub fn acos(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::acos)
}

#[pyfunction]
pub fn atan(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::atan)
}

#[pyfunction]
pub fn atanh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::atanh)
}

#[pyfunction]
pub fn asinh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::asinh)
}

#[pyfunction]
pub fn acosh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::acosh)
}

#[pyfunction]
pub fn expm1(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::expm1)
}

#[pyfunction]
pub fn log1p(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::log1p)
}

#[pyfunction]
pub fn sinh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::sinh)
}

#[pyfunction]
pub fn cosh(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::cosh)
}

#[pyfunction]
pub fn log2(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::log2)
}

#[pyfunction]
pub fn log10(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::log10)
}

#[pyfunction]
pub fn exp2(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::exp2)
}

#[pyfunction]
pub fn abs(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::abs)
}

#[pyfunction]
pub fn sqrt(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::sqrt)
}

#[pyfunction]
pub fn neg(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::neg)
}

#[pyfunction]
pub fn recip(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::recip)
}

#[pyfunction]
pub fn sign(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::sign)
}

#[pyfunction]
pub fn floor(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::floor)
}

#[pyfunction]
pub fn ceil(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::ceil)
}

#[pyfunction]
pub fn round(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::round)
}

#[pyfunction]
pub fn trunc(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::trunc)
}

#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::clamp(x, min_val, max_val))
}

#[pyfunction]
pub fn scalar_mul(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::scalar_mul(x, scalar))
}

#[pyfunction]
pub fn scalar_add(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::scalar_add(x, scalar))
}

#[pyfunction]
pub fn scalar_div(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::scalar_div(x, scalar))
}

#[pyfunction]
pub fn scalar_sub(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::scalar_sub(x, scalar))
}

#[pyfunction]
pub fn pow(input: &PyTensor, exp: f64, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, |x| coeus_autograd::pow(x, exp))
}

/// Element-wise remainder (`torch.remainder`): `a - floor(a / b) * b`,
/// carrying the sign of the divisor. Gradient flows to `a` (identity) and
/// `b` (`-floor(a / b)`).
#[pyfunction]
pub fn remainder(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_binary(a, b, py, coeus_autograd::remainder)
}

/// Element-wise maximum (`torch.maximum`, Burn `Tensor::max_pair`). Gradient
/// routes to the larger operand; ties resolve to `a`.
#[pyfunction]
pub fn maximum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_binary(a, b, py, coeus_autograd::maximum)
}

/// Element-wise minimum (`torch.minimum`, Burn `Tensor::min_pair`). Gradient
/// routes to the smaller operand; ties resolve to `a`.
#[pyfunction]
pub fn minimum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_binary(a, b, py, coeus_autograd::minimum)
}

#[pyfunction]
pub fn sin(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::sin)
}

#[pyfunction]
pub fn cos(input: &PyTensor, py: Python<'_>) -> PyTensor {
    apply_unary(input, py, coeus_autograd::cos)
}
