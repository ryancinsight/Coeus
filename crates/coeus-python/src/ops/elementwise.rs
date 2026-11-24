use crate::{error::map_backend_error, tensor::PyTensor};
use coeus_autograd::Var;
use pyo3::prelude::*;

/// Generate the module functions and tensor methods that share one autograd
/// unary kernel and one GIL/error dispatch path.
macro_rules! unary_elementwise_ops {
    ($(($name:ident, $op:path $(, $doc:expr)?)),+ $(,)?) => {
        $(
            $(#[doc = $doc])?
            #[pyfunction]
            pub fn $name(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
                input.unary_dispatch(py, $op)
            }
        )+

        #[pymethods]
        impl PyTensor {
            $(
                $(#[doc = $doc])?
                fn $name(&self, py: Python<'_>) -> PyResult<Self> {
                    self.unary_dispatch(py, $op)
                }
            )+
        }
    };
}

unary_elementwise_ops!(
    (exp, coeus_autograd::exp),
    (log, coeus_autograd::log),
    (
        erf,
        coeus_autograd::erf,
        "Gauss error function (`torch.erf`); differentiable, d/dx = 2/√π·e^(−x²)."
    ),
    (
        erfc,
        coeus_autograd::erfc,
        "Complementary error function (`torch.special.erfc`); differentiable."
    ),
    (tan, coeus_autograd::tan),
    (asin, coeus_autograd::asin),
    (acos, coeus_autograd::acos),
    (atan, coeus_autograd::atan),
    (atanh, coeus_autograd::atanh),
    (asinh, coeus_autograd::asinh),
    (acosh, coeus_autograd::acosh),
    (expm1, coeus_autograd::expm1),
    (log1p, coeus_autograd::log1p),
    (sinh, coeus_autograd::sinh),
    (cosh, coeus_autograd::cosh),
    (log2, coeus_autograd::log2),
    (log10, coeus_autograd::log10),
    (exp2, coeus_autograd::exp2),
    (abs, coeus_autograd::abs),
    (sqrt, coeus_autograd::sqrt),
    (recip, coeus_autograd::recip),
    (sign, coeus_autograd::sign),
    (floor, coeus_autograd::floor),
    (ceil, coeus_autograd::ceil),
    (round, coeus_autograd::round),
    (trunc, coeus_autograd::trunc),
    (sin, coeus_autograd::sin),
    (cos, coeus_autograd::cos),
);

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
    let tensor = tensor.map_err(map_backend_error)?;
    Var::new(tensor, false)
        .map(PyTensor::from_var)
        .map_err(map_backend_error)
}

/// Alias matching `torch.lgamma`.
#[pyfunction]
pub fn lgamma(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    gammaln(input, py)
}

#[pyfunction]
pub fn neg(input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    input.unary_dispatch(py, coeus_autograd::neg)
}

#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<PyTensor> {
    input.unary_dispatch(py, |v| coeus_autograd::clamp(v, min_val, max_val))
}

#[pyfunction]
pub fn scalar_mul(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_add(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_add(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_div(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_div(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn scalar_sub(input: &PyTensor, scalar: f64, py: Python<'_>) -> PyResult<PyTensor> {
    let inner = py.allow_threads(|| coeus_autograd::scalar_sub(&input.inner, scalar));
    inner
        .map(|inner| PyTensor { inner })
        .map_err(map_backend_error)
}

#[pyfunction]
pub fn pow(input: &PyTensor, exp: f64, py: Python<'_>) -> PyResult<PyTensor> {
    input.unary_dispatch(py, |v| coeus_autograd::pow(v, exp))
}

/// Element-wise remainder (`torch.remainder`): `a - floor(a / b) * b`,
/// carrying the sign of the divisor. Gradient flows to `a` (identity) and
/// `b` (`-floor(a / b)`).
#[pyfunction]
pub fn remainder(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    a.binary_dispatch(py, b, coeus_autograd::remainder)
}

/// Element-wise maximum (`torch.maximum`, Burn `Tensor::max_pair`). Gradient
/// routes to the larger operand; ties resolve to `a`.
#[pyfunction]
pub fn maximum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    a.binary_dispatch(py, b, coeus_autograd::maximum)
}

/// Element-wise minimum (`torch.minimum`, Burn `Tensor::min_pair`). Gradient
/// routes to the smaller operand; ties resolve to `a`.
#[pyfunction]
pub fn minimum(a: &PyTensor, b: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
    a.binary_dispatch(py, b, coeus_autograd::minimum)
}
