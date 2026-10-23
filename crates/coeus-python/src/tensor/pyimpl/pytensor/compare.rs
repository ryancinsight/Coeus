//! Elementwise comparisons, plus the grad/detach surface and the data setter.

use super::CompareOp;
use super::PyTensor;
use crate::error::map_backend_error;
use coeus_autograd::Var;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn eq(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Eq)
    }

    fn lt(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Lt)
    }

    fn gt(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Gt)
    }

    fn ne(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Ne)
    }

    fn ge(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Ge)
    }

    fn le(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        self.comparison_dispatch(py, other, CompareOp::Le)
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
