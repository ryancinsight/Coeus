//! In-place mutation of the wrapped tensor.
//!
//! These mutators follow torch's in-place contract: the receiver is rewritten
//! and handed back as the same Python object, so `t.zero_() is t` holds. A
//! tensor whose value autograd is tracking is refused instead of rewritten
//! (see [`PyTensor::reject_tracked_mutation`]).

use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

impl PyTensor {
    /// Overwrite every element with `value`, keeping the shape.
    ///
    /// # Errors
    /// Propagates a tensor-construction failure.
    fn fill_all(&mut self, value: f64) -> PyResult<()> {
        let shape = self.inner.tensor.shape().to_vec();
        let numel: usize = shape.iter().product();
        let data = vec![value; numel];
        self.inner.tensor =
            coeus_tensor::Tensor::from_slice(shape, &data).map_err(map_backend_error)?;
        Ok(())
    }
}

/// Shared body of `fill_`/`zero_`/`one_`: guard, fill, and return the receiver.
///
/// The borrow is dropped before the handle is cloned, so the caller gets its
/// own object back rather than a fresh `PyTensor` wrapping copied data.
fn fill_receiver<'py>(slf: &Bound<'py, PyTensor>, value: f64, op: &str) -> PyResult<Py<PyTensor>> {
    {
        let mut this = slf.borrow_mut();
        this.reject_tracked_mutation(op)?;
        this.fill_all(value)?;
    }
    Ok(slf.clone().unbind())
}

#[pymethods]
impl PyTensor {
    fn fill_<'py>(slf: &Bound<'py, Self>, value: f64) -> PyResult<Py<Self>> {
        fill_receiver(slf, value, "fill_")
    }

    fn zero_<'py>(slf: &Bound<'py, Self>) -> PyResult<Py<Self>> {
        fill_receiver(slf, 0.0, "zero_")
    }

    fn one_<'py>(slf: &Bound<'py, Self>) -> PyResult<Py<Self>> {
        fill_receiver(slf, 1.0, "one_")
    }

    fn __iadd__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        self.reject_tracked_mutation("__iadd__")?;
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            let a = self.inner.tensor.clone();
            let b = other.inner.tensor.clone();
            coeus_ops::add(&a, &b, &backend)
        });
        self.inner.tensor = new_t.map_err(map_backend_error)?;
        Ok(())
    }

    fn __isub__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        self.reject_tracked_mutation("__isub__")?;
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            coeus_ops::sub(&self.inner.tensor, &other.inner.tensor, &backend)
        });
        self.inner.tensor = new_t.map_err(map_backend_error)?;
        Ok(())
    }

    fn __imul__(&mut self, other: &PyTensor, py: Python<'_>) -> PyResult<()> {
        self.reject_tracked_mutation("__imul__")?;
        let new_t = py.allow_threads(|| {
            let backend = coeus_core::MoiraiBackend::new();
            coeus_ops::mul(&self.inner.tensor, &other.inner.tensor, &backend)
        });
        self.inner.tensor = new_t.map_err(map_backend_error)?;
        Ok(())
    }
}
