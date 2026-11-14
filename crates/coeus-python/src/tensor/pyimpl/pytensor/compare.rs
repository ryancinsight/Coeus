//! Elementwise comparisons, plus the grad/detach surface and the data setter.

use super::dispatch::CompareOp;
use super::PyTensor;
use crate::error::map_backend_error;
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__eq__", other, CompareOp::Eq)
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__lt__", other, CompareOp::Lt)
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__gt__", other, CompareOp::Gt)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__ne__", other, CompareOp::Ne)
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__ge__", other, CompareOp::Ge)
    }

    fn __le__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, "__le__", other, CompareOp::Le)
    }

    /// Tensor method form for equality comparison.
    fn eq(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__eq__(other, py)
    }

    /// Tensor method form for less-than comparison.
    fn lt(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__lt__(other, py)
    }

    /// Tensor method form for greater-than comparison.
    fn gt(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__gt__(other, py)
    }

    /// Tensor method form for inequality comparison.
    fn ne(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__ne__(other, py)
    }

    /// Tensor method form for greater-or-equal comparison.
    fn ge(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__ge__(other, py)
    }

    /// Tensor method form for less-or-equal comparison.
    fn le(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
        self.__le__(other, py)
    }

    fn backward(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.backward())
            .map_err(map_backend_error)
    }

    fn detach(&self) -> PyResult<Self> {
        Var::new(self.inner.tensor.clone(), false)
            .map(Self::from_var)
            .map_err(map_backend_error)
    }

    /// Toggle gradient tracking on this tensor and return the receiver.
    ///
    /// Unlike [`Self::detach`], which copies the data into a new untracked
    /// tensor, this rewrites the receiver's own autograd state, so it returns
    /// the same Python object (`t.requires_grad_(False) is t`) the way torch's
    /// in-place setters do. The flag is only rebuilt when it actually changes,
    /// so re-asserting the current mode does not disturb the existing
    /// gradient buffer.
    ///
    /// # Errors
    /// Propagates a backend failure while re-wrapping the tensor.
    fn requires_grad_<'py>(slf: &Bound<'py, Self>, requires_grad: bool) -> PyResult<Py<Self>> {
        {
            let mut this = slf.borrow_mut();
            if requires_grad != this.inner.grad.is_some() {
                let tensor = this.inner.tensor.clone();
                this.inner = Var::new(tensor, requires_grad).map_err(map_backend_error)?;
            }
        }
        Ok(slf.clone().unbind())
    }

    /// Zero the gradient of this tensor.
    pub fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad().map_err(map_backend_error)
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
        self.inner.tensor =
            coeus_tensor::Tensor::from_slice(shape, &data).map_err(map_backend_error)?;
        Ok(())
    }
}
