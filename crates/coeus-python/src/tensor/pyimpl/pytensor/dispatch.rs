//! Operator tags and the GIL-crossing helper methods the `#[pymethods]`
//! sections call.
//!
//! Each helper copies its operands out of the `PyTensor` wrapper before
//! `Python::allow_threads`, so only `Send` data crosses the GIL boundary and
//! the released closure is a small tag match rather than a captured kernel.

use crate::error::map_backend_error;
use coeus_autograd::Var;
use pyo3::prelude::*;

use super::PyTensor;

/// Closed-set dispatch tag for Python binary operators.
///
/// The dispatcher in [`PyTensor::binop_dispatch`] walks this enum rather
/// than carrying a kernel closure across `py.allow_threads` so the
/// generated inner closure is `Ungil + Send` (the GIL-release point
/// does not permit arbitrary captured closures).
#[derive(Copy, Clone, Debug)]
pub(super) enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Closed-set dispatch tag for Python comparison operators.
#[derive(Copy, Clone, Debug)]
pub(super) enum CompareOp {
    Eq,
    Lt,
    Gt,
    Ne,
    Ge,
    Le,
}

impl PyTensor {
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
    pub(super) fn binop_dispatch<'py>(
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
            inner.map(Self::from_var).map_err(map_backend_error)
        } else if let Ok(scalar) = other.extract::<f64>() {
            let inner = py.allow_threads(|| match op {
                BinOp::Add => coeus_autograd::scalar_add(&self_inner, scalar),
                BinOp::Sub => coeus_autograd::scalar_sub(&self_inner, scalar),
                BinOp::Mul => coeus_autograd::scalar_mul(&self_inner, scalar),
                BinOp::Div => coeus_autograd::scalar_div(&self_inner, scalar),
            });
            inner.map(Self::from_var).map_err(map_backend_error)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{op_label}: unsupported operand type for Tensor and {}",
                other.get_type().name()?,
            )))
        }
    }

    #[inline]
    pub(crate) fn unary_dispatch<F>(&self, py: Python<'_>, op: F) -> PyResult<Self>
    where
        F: FnOnce(&Var<f64>) -> Result<Var<f64>, coeus_core::BackendError> + Send,
    {
        let inner = py.allow_threads(|| op(&self.inner));
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    #[inline]
    pub(crate) fn binary_dispatch<F>(
        &self,
        py: Python<'_>,
        other: &PyTensor,
        op: F,
    ) -> PyResult<Self>
    where
        F: FnOnce(&Var<f64>, &Var<f64>) -> Result<Var<f64>, coeus_core::BackendError> + Send,
    {
        // Pre-clone both operands so only `Send` data crosses the GIL boundary.
        let x = self.inner.clone();
        let y = other.inner.clone();
        let inner = py.allow_threads(move || op(&x, &y));
        inner.map(Self::from_var).map_err(map_backend_error)
    }

    #[inline]
    pub(super) fn comparison_dispatch(
        &self,
        py: Python<'_>,
        other: &PyTensor,
        op: CompareOp,
    ) -> PyResult<Self> {
        let inner = py.allow_threads(|| match op {
            CompareOp::Eq => coeus_autograd::eq(&self.inner, &other.inner),
            CompareOp::Lt => coeus_autograd::lt(&self.inner, &other.inner),
            CompareOp::Gt => coeus_autograd::gt(&self.inner, &other.inner),
            CompareOp::Ne => coeus_autograd::ne(&self.inner, &other.inner),
            CompareOp::Ge => coeus_autograd::ge(&self.inner, &other.inner),
            CompareOp::Le => coeus_autograd::le(&self.inner, &other.inner),
        });
        inner.map(|inner| Self { inner }).map_err(map_backend_error)
    }
}
