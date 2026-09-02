// ── PyTensor struct definition and single #[pymethods] block ──

use coeus_autograd::Var;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Closed-set dispatch tag for Python binary operators.
///
/// The dispatcher in `PyTensor::binop_dispatch` walks this enum rather
/// than carrying a kernel closure across `py.allow_threads` so the
/// generated inner closure is `Ungil + Send` (the GIL-release point
/// does not permit arbitrary captured closures).
#[derive(Copy, Clone, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Python-exposed tensor class wrapping autograd variables.
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    /// Underlying Rust autograd variable holding the tensor data and gradient.
    pub inner: Var<f64>,
}

impl PyTensor {
    pub(crate) fn from_var(inner: Var<f64>) -> Self {
        Self {
            inner: crate::grad_mode::maybe_untrack_var(inner),
        }
    }

    /// Dispatch a Python binary-operator invocation to either the tensor
    /// kernel or the scalar broadcast kernel, depending on whether
    /// `other` is a Python `Tensor` or a Python number.
    ///
    /// Used by `__add__`, `__sub__`, `__mul__`, `__truediv__` to give
    /// Pycoeus tensors the same scalar-arithmetic ergonomics as
    /// PyTorch / JAX / MLX: `t - 1.0`, `t / 2.0`, etc.
    ///
    /// Pre-clones the `inner: Var<f64>` and the scalar value before
    /// `py.allow_threads(...)` to carry only `Send` data across the
    /// GIL boundary; the closure inside `allow_threads` is a small
    /// `match` on the `op` tag rather than a captured generic kernel.
    #[inline]
    fn binop_dispatch<'py>(
        &self,
        py: Python<'py>,
        op_label: &str,
        other: &Bound<'py, PyAny>,
        op: BinOp,
    ) -> PyResult<Self> {
        let self_inner = self.inner.clone();
        if let Ok(t) = other.downcast::<PyTensor>() {
            let other_inner = t.borrow().inner.clone();
            let inner = py.allow_threads(|| match op {
                BinOp::Add => coeus_autograd::add(&self_inner, &other_inner),
                BinOp::Sub => coeus_autograd::sub(&self_inner, &other_inner),
                BinOp::Mul => coeus_autograd::mul(&self_inner, &other_inner),
                BinOp::Div => coeus_autograd::div(&self_inner, &other_inner),
            });
            Ok(Self::from_var(inner))
        } else if let Ok(scalar) = other.extract::<f64>() {
            let inner = py.allow_threads(|| match op {
                BinOp::Add => coeus_autograd::scalar_add(&self_inner, scalar),
                BinOp::Sub => coeus_autograd::scalar_sub(&self_inner, scalar),
                BinOp::Mul => coeus_autograd::scalar_mul(&self_inner, scalar),
                BinOp::Div => coeus_autograd::scalar_div(&self_inner, scalar),
            });
            Ok(Self::from_var(inner))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{op_label}: unsupported operand type for Tensor and {}",
                other.get_type().name()?,
            )))
        }
    }
}

#[pymethods]
impl PyTensor {
    // ── Constructor + basic properties ──

    #[new]
    #[pyo3(signature = (data, shape = None, requires_grad = false))]
    fn new(data: Vec<f64>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let tensor = coeus_tensor::Tensor::from_slice(shape, &data);
        Ok(Self {
            inner: Var::new(tensor, requires_grad),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.tensor.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> Vec<f64> {
        self.inner.tensor.to_contiguous().as_slice().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.tensor.ndim()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.grad.is_some()
    }

    fn item(&self) -> PyResult<f64> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item(): tensor has {numel} elements, expected 1"
            )));
        }
        let contiguous = self.inner.tensor.to_contiguous();
        Ok(contiguous.as_slice()[0])
    }

    fn tolist(&self) -> Vec<f64> {
        self.data()
    }

    fn clone_tensor(&self) -> Self {
        self.clone()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.tensor.is_contiguous()
    }

    #[pyo3(name = "numel")]
    fn numel_method(&self) -> usize {
        self.inner.tensor.shape().iter().product()
    }

    // ── Python protocol / dunder methods ──

    fn __repr__(&self) -> String {
        let shape = self.shape();
        let requires_grad = self.inner.grad.is_some();
        let data = self.inner.tensor.to_contiguous();
        let vals = data.as_slice();
        let max_display = 8;
        let data_str = if vals.is_empty() {
            "[]".to_string()
        } else if vals.len() <= max_display {
            let formatted: Vec<String> = vals
                .iter()
                .map(|&v| {
                    if v == (v as i64) as f64 && v.abs() < 1e6 {
                        format!("{:.1}", v)
                    } else {
                        format!("{:.4}", v)
                    }
                })
                .collect();
            format!("[{}]", formatted.join(", "))
        } else {
            let first: Vec<String> = vals[..3].iter().map(|&v| format!("{:.4}", v)).collect();
            let last: Vec<String> = vals[vals.len() - 2..]
                .iter()
                .map(|&v| format!("{:.4}", v))
                .collect();
            format!("[{}, ..., {}]", first.join(", "), last.join(", "))
        };
        if requires_grad {
            format!("Tensor({data_str}, shape={shape:?}, requires_grad=True)")
        } else {
            format!("Tensor({data_str}, shape={shape:?})")
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __bool__(&self) -> PyResult<bool> {
        let numel: usize = self.inner.tensor.shape().iter().product();
        if numel == 0 {
            return Ok(false);
        }
        if numel != 1 {
            return Ok(true);
        }
        Ok(self.inner.tensor.to_contiguous().as_slice()[0] != 0.0)
    }

    fn __float__(&self) -> PyResult<f64> {
        self.item()
    }

    fn __int__(&self) -> PyResult<i64> {
        self.item().map(|v| v as i64)
    }

    fn __len__(&self) -> PyResult<usize> {
        let shape = self.inner.tensor.shape();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__len__: tensor is 0-dimensional",
            ));
        }
        Ok(shape[0])
    }

    // ── Arithmetic operators ──
    //
    // All five binary operators (`+`, `-`, `*`, `/`, `@`) accept either a
    // `Tensor` (elementwise) or a Python `float` scalar (broadcast via
    // `coeus_autograd::scalar_*`). The runtime discrimination is at the
    // PyO3 boundary (single arity per Python method), so the operator
    // wrappers take `&Bound<'_, PyAny>` and route to the right kernel.
    //
    // The reflected-side (e.g. `float + Tensor`) wrappers (`__radd__`,
    // `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`) receive only
    // scalars because Python's binary-op dispatch sends a `Tensor` to
    // both sides' reflected methods when the LHS-typed op fails; those
    // operators only enter when the LHS has no `+`/`*` method or
    // returned NotImplemented, so the reflected overloads are guaranteed
    // to be invoked with a non-Tensor LHS.

    fn __add__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__add__", other, BinOp::Add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__sub__", other, BinOp::Sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__mul__", other, BinOp::Mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.binop_dispatch(py, "__truediv__", other, BinOp::Div)
    }

    fn __matmul__(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::matmul(&self.inner, &other.inner));
        Ok(Self::from_var(inner))
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::neg(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn __abs__(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::abs(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn __rmul__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&self.inner, scalar));
        Ok(Self::from_var(inner))
    }

    fn __radd__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_add(&self.inner, scalar));
        Ok(Self::from_var(inner))
    }

    fn __rsub__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        // `scalar - self`: negate, then add the scalar.
        let inner = py.allow_threads(|| {
            let neg = coeus_autograd::neg(&self.inner);
            coeus_autograd::scalar_add(&neg, scalar)
        });
        Ok(Self::from_var(inner))
    }

    fn __rtruediv__(&self, scalar: f64, py: Python<'_>) -> PyResult<Self> {
        // `scalar / self`: `scalar * reciprocal(self)`.
        let inner = py.allow_threads(|| {
            let recip = coeus_autograd::recip(&self.inner);
            coeus_autograd::scalar_mul(&recip, scalar)
        });
        Ok(Self::from_var(inner))
    }

    fn __rpow__(&self, base: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        // `base ** self`: exp(self * ln(base)).
        let inner = py.allow_threads(|| {
            let scaled = coeus_autograd::scalar_mul(&self.inner, base.ln());
            coeus_autograd::exp(&scaled)
        });
        Ok(Self::from_var(inner))
    }

    // ── Unary math ops ──

    fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::exp(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn erf(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::erf(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn erfc(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::erfc(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::tan(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn asin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::asin(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn acos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::acos(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn atan(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::atan(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn atanh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::atanh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn asinh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::asinh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn acosh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::acosh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn expm1(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::expm1(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log1p(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log1p(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sinh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sinh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn cosh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cosh(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log2(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log10(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn exp2(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::exp2(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn selu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::selu(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn log(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::abs(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sqrt(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn recip(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::recip(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sign(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::floor(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::ceil(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn round(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::round(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn trunc(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::trunc(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sin(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cos(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn pow(&self, exp: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self::from_var(inner))
    }

    fn __pow__(&self, exp: f64, _modulo: Option<i64>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::pow(&self.inner, exp));
        Ok(Self::from_var(inner))
    }

    fn clamp(&self, min_val: f64, max_val: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::clamp(&self.inner, min_val, max_val));
        Ok(Self::from_var(inner))
    }

    fn scale(&self, s: f64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::scalar_mul(&self.inner, s));
        Ok(Self::from_var(inner))
    }

    // ── Reduction ops ──

    fn sum_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sum_axis(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn mean_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::mean_axis(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn sum(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::sum(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn mean(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::mean(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn softmax(&self, dim: i64, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::softmax(&self.inner, dim as isize));
        Ok(Self::from_var(inner))
    }

    fn log_softmax(&self, dim: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log_softmax(&self.inner, dim));
        Ok(Self::from_var(inner))
    }

    fn cumsum(&self, dim: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::cumsum(&self.inner, dim));
        Ok(Self::from_var(inner))
    }

    fn max_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::max_axis(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn min_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::min_axis(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn log_sum_exp(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::log_sum_exp(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    // ── Shape manipulation ──

    fn reshape(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::reshape(&self.inner, shape));
        Ok(Self::from_var(inner))
    }

    fn permute(&self, dims: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::permute(&self.inner, &dims));
        Ok(Self::from_var(inner))
    }

    #[pyo3(signature = (axis = None))]
    fn squeeze(&self, axis: Option<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::squeeze(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn unsqueeze(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::unsqueeze(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn t(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::transpose_2d(&self.inner));
        Ok(Self::from_var(inner))
    }

    fn transpose(&self, dim0: usize, dim1: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::transpose(&self.inner, dim0, dim1));
        Ok(Self::from_var(inner))
    }

    fn contiguous(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::contiguous(&self.inner));
        Ok(Self::from_var(inner))
    }

    #[pyo3(signature = (start_dim = 0, end_dim = -1))]
    fn flatten(&self, start_dim: i64, end_dim: i64, py: Python<'_>) -> PyResult<Self> {
        let ndim = self.inner.tensor.ndim() as i64;
        let start = if start_dim < 0 {
            (ndim + start_dim).max(0)
        } else {
            start_dim
        } as usize;
        let end = if end_dim < 0 {
            (ndim + end_dim).max(0)
        } else {
            end_dim.min(ndim - 1)
        } as usize;
        let shape = self.inner.tensor.shape();
        let mut new_shape: Vec<usize> = shape[..start].to_vec();
        new_shape.push(shape[start..=end].iter().product());
        new_shape.extend_from_slice(&shape[end + 1..]);
        let inner = py.allow_threads(|| coeus_autograd::reshape(&self.inner, new_shape));
        Ok(Self::from_var(inner))
    }

    fn view(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::reshape(&self.inner, shape));
        Ok(Self::from_var(inner))
    }

    fn expand(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        let src = self.inner.tensor.shape().to_vec();
        if src.len() != shape.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expand: ndim mismatch: src={} target={}",
                src.len(),
                shape.len()
            )));
        }
        for (s, t) in src.iter().zip(shape.iter()) {
            if *s != 1 && *s != *t {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "expand: incompatible dim: src={s} target={t}"
                )));
            }
        }
        let inner = py.allow_threads(|| {
            let zeros_v = Var::new(
                coeus_tensor::Tensor::<f64, coeus_core::MoiraiBackend>::zeros(shape),
                false,
            );
            coeus_autograd::add(&self.inner, &zeros_v)
        });
        Ok(Self::from_var(inner))
    }

    fn broadcast_to(&self, shape: Vec<usize>, py: Python<'_>) -> PyResult<Self> {
        self.expand(shape, py)
    }

    fn flip(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        let inner = py.allow_threads(|| coeus_autograd::flip(&self.inner, axis));
        Ok(Self::from_var(inner))
    }

    fn repeat(&self, reps: Vec<usize>, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::tile(&self.inner, &reps));
        Self::from_var(inner)
    }

    #[getter]
    #[expect(non_snake_case, reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097")]
    fn T(&self, py: Python<'_>) -> PyResult<Self> {
        let ndim = self.inner.tensor.ndim();
        if ndim != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tensor.T: expected 2-D tensor, got {ndim}-D"
            )));
        }
        let inner = py.allow_threads(|| coeus_autograd::permute(&self.inner, &[1, 0]));
        Ok(Self::from_var(inner))
    }

    // ── Indexing / slicing ──

    fn __getitem__(&self, index: &pyo3::Bound<'_, pyo3::PyAny>, py: Python<'_>) -> PyResult<Self> {
        let shape = self.inner.tensor.shape();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "__getitem__: cannot index a 0-dimensional tensor",
            ));
        }

        if let Ok(i) = index.extract::<i64>() {
            let n = shape[0] as i64;
            let normalized = if i < 0 { n + i } else { i };
            if !(0..n).contains(&normalized) {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__getitem__: index {i} out of range for dim 0 size {n}"
                )));
            }
            let idx = normalized as usize;
            let ranges: Vec<(usize, usize)> = shape
                .iter()
                .enumerate()
                .map(|(d, &s)| if d == 0 { (idx, idx + 1) } else { (0, s) })
                .collect();
            let inner = py.allow_threads(|| {
                let sliced = coeus_autograd::slice(&self.inner, &ranges);
                coeus_autograd::squeeze(&sliced, Some(0))
            });
            return Ok(Self::from_var(inner));
        }

        if let Ok(sl) = index.downcast::<pyo3::types::PySlice>() {
            let n = shape[0];
            let slice_len = isize::try_from(n).map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(format!(
                    "__getitem__: dim 0 size {n} exceeds Python slice bounds"
                ))
            })?;
            let indices = sl.indices(slice_len)?;
            if indices.step != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "__getitem__: slice step {} is unsupported; expected 1",
                    indices.step
                )));
            }
            let start = indices.start.max(0) as usize;
            let stop = (indices.stop.max(0) as usize).min(n);
            if start >= stop {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "__getitem__: empty slice [{start}:{stop}]"
                )));
            }
            let ranges: Vec<(usize, usize)> = shape
                .iter()
                .enumerate()
                .map(|(d, &s)| if d == 0 { (start, stop) } else { (0, s) })
                .collect();
            let inner = py.allow_threads(|| coeus_autograd::slice(&self.inner, &ranges));
            return Ok(Self::from_var(inner));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "__getitem__: index must be an int or a slice",
        ))
    }

    fn __setitem__(
        &mut self,
        index: &pyo3::Bound<'_, pyo3::PyAny>,
        value: &pyo3::Bound<'_, pyo3::PyAny>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let shape = self.inner.tensor.shape().to_vec();
        if shape.is_empty() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "__setitem__: cannot index a 0-dimensional tensor",
            ));
        }
        let n = shape[0];

        let idx = if let Ok(i) = index.extract::<i64>() {
            let normalized = if i < 0 { n as i64 + i } else { i };
            if normalized < 0 || normalized as usize >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "__setitem__: index {i} out of range for dim 0 size {n}"
                )));
            }
            normalized as usize
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__setitem__: index must be an int",
            ));
        };

        let row_numel: usize = shape[1..].iter().product::<usize>().max(1);
        let fill_data: Vec<f64> = if let Ok(v) = value.extract::<f64>() {
            vec![v; row_numel]
        } else if let Ok(t) = value.extract::<PyTensor>() {
            let cont = t.inner.tensor.to_contiguous();
            cont.as_slice().to_vec()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "__setitem__: value must be a float or Tensor",
            ));
        };

        if fill_data.len() != row_numel {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "__setitem__: value has {} elements but row requires {}",
                fill_data.len(),
                row_numel
            )));
        }

        let _ = py;
        let numel: usize = shape.iter().product();
        let mut host = vec![0.0f64; numel];
        use coeus_core::ComputeBackend;
        let backend = coeus_core::MoiraiBackend::new();
        backend.copy_to_host(self.inner.tensor.storage(), &mut host);
        let start = idx * row_numel;
        host[start..start + row_numel].copy_from_slice(&fill_data);
        self.inner.tensor = coeus_tensor::Tensor::from_slice(shape, &host);
        Ok(())
    }

    fn __iter__(&self) -> PyResult<crate::tensor::PyTensorIterator> {
        let length = self.inner.tensor.shape().first().copied().ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("__iter__: tensor is 0-dimensional")
        })?;
        Ok(crate::tensor::PyTensorIterator {
            tensor: self.clone(),
            current: 0,
            length,
        })
    }

    // ── Comparison ops ──

    fn eq(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::eq(&self.inner, &other.inner));
        Self { inner }
    }

    fn lt(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::lt(&self.inner, &other.inner));
        Self { inner }
    }

    fn gt(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::gt(&self.inner, &other.inner));
        Self { inner }
    }

    fn ne(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::ne(&self.inner, &other.inner));
        Self { inner }
    }

    fn ge(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::ge(&self.inner, &other.inner));
        Self { inner }
    }

    fn le(&self, other: &PyTensor, py: Python<'_>) -> Self {
        let inner = py.allow_threads(|| coeus_autograd::le(&self.inner, &other.inner));
        Self { inner }
    }

    // ── Grad utilities ──

    fn backward(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.backward())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn detach(&self) -> Self {
        Self {
            inner: Var::new(self.inner.tensor.clone(), false),
        }
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> Self {
        if requires_grad && self.inner.grad.is_none() {
            let t = self.inner.tensor.clone();
            self.inner = Var::new(t, true);
        } else if !requires_grad && self.inner.grad.is_some() {
            let t = self.inner.tensor.clone();
            self.inner = Var::new(t, false);
        }
        self.clone()
    }

    /// Zero the gradient of this tensor.
    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    #[setter]
    fn set_data(&mut self, data: Vec<f64>) -> PyResult<()> {
        let shape = self.inner.tensor.shape().to_vec();
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data length {} does not match tensor shape {:?} (expected {})",
                data.len(),
                shape,
                expected_len
            )));
        }
        self.inner.tensor = coeus_tensor::Tensor::from_slice(shape, &data);
        Ok(())
    }

    // ── In-place mutation ──

    fn fill_(&mut self, value: f64) -> Self {
        let shape = self.inner.tensor.shape().to_vec();
        let numel: usize = shape.iter().product();
        let data = vec![value; numel];
        self.inner.tensor = coeus_tensor::Tensor::from_slice(shape, &data);
        self.clone()
    }

    fn zero_(&mut self) -> Self {
        self.fill_(0.0)
    }

    fn one_(&mut self) -> Self {
        self.fill_(1.0)
    }

    fn __iadd__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            let a = self.inner.tensor.clone();
            let b = other.inner.tensor.clone();
            coeus_ops::add(&a, &b, &backend)
        });
        self.inner.tensor = new_t;
        Ok(())
    }

    fn __isub__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            coeus_ops::sub(&self.inner.tensor, &other.inner.tensor, &backend)
        });
        self.inner.tensor = new_t;
        Ok(())
    }

    fn __imul__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            coeus_ops::mul(&self.inner.tensor, &other.inner.tensor, &backend)
        });
        self.inner.tensor = new_t;
        Ok(())
    }

    // ── Dtype cast ──

    fn float(&self) -> Self {
        self.clone()
    }

    fn double(&self) -> Self {
        self.clone()
    }

    fn long(&self) -> Self {
        let data: Vec<f64> = self
            .inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|&v| (v as i64) as f64)
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data);
        Self {
            inner: Var::new(t, false),
        }
    }

    fn int(&self) -> Self {
        self.long()
    }

    fn half(&self) -> Self {
        let data: Vec<f64> = self
            .inner
            .tensor
            .to_contiguous()
            .as_slice()
            .iter()
            .map(|&v| f64::from(eunomia::F16::from_f64(v).to_f32()))
            .collect();
        let shape = self.inner.tensor.shape().to_vec();
        let t = coeus_tensor::Tensor::from_slice(shape, &data);
        Self {
            inner: Var::new(t, false),
        }
    }

    fn to(&self, dtype: &str) -> PyResult<Self> {
        match dtype {
            "float" | "float32" | "float64" | "double" => Ok(self.float()),
            "long" | "int64" => Ok(self.long()),
            "int" | "int32" => Ok(self.int()),
            "half" | "float16" => Ok(self.half()),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to: unknown dtype '{other}'; supported: float, double, long, int, half, float16, float32, float64, int32, int64"
            ))),
        }
    }

    fn type_as(&self, _other: &PyTensor) -> Self {
        self.clone()
    }

    // ── Grad getter (must stay in pymethods for PyO3 getter) ──

    #[getter]
    fn grad(&self) -> Option<Vec<f64>> {
        self.inner
            .grad()
            .map(|g| g.to_contiguous().as_slice().to_vec())
    }
}
