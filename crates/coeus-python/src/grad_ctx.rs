//! No-gradient context manager for Python (PyO3).
//!
//! Provides the `no_grad` context manager that suppresses autograd within
//! its scope, mirroring PyTorch's `torch.no_grad()`.

use pyo3::prelude::*;

use crate::grad_mode;

/// Context manager that disables gradient tracking within its scope.
///
/// Usage:
/// ```python
/// with pycoeus.no_grad():
///     y = model(x)   # no gradients computed
/// ```
#[pyclass(name = "no_grad")]
#[derive(Default)]
pub struct NoGradCtx {
    active: std::sync::atomic::AtomicBool,
}

#[pymethods]
impl NoGradCtx {
    /// Create a new no-grad context manager.
    #[new]
    pub fn new() -> Self {
        Self {
            active: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Enter the no-grad context, suppressing gradient tracking.
    pub fn __enter__(&self) {
        if !self.active.swap(true, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::push_no_grad();
        }
    }

    /// Exit the no-grad context, restoring gradient tracking.
    pub fn __exit__(
        &self,
        _exc_type: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_val: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_tb: pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        if self.active.swap(false, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::pop_no_grad();
        }
        false
    }
}

impl Drop for NoGradCtx {
    fn drop(&mut self) {
        if self.active.swap(false, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::pop_no_grad();
        }
    }
}
