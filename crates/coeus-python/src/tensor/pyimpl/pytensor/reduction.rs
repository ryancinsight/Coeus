//! Whole-tensor and per-axis reductions.

use super::PyTensor;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    fn sum_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::sum_axis(x, axis))
    }

    fn mean_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::mean_axis(x, axis))
    }

    fn sum(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::sum)
    }

    fn mean(&self, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, coeus_autograd::mean)
    }

    fn softmax(&self, dim: i64, py: Python<'_>) -> PyResult<Self> {
        let axis = isize::try_from(dim).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "softmax: dim {dim} exceeds the axis range"
            ))
        })?;
        self.unary_dispatch(py, |x| coeus_autograd::softmax(x, axis))
    }

    fn log_softmax(&self, dim: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::log_softmax(x, dim))
    }

    fn cumsum(&self, dim: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::cumsum(x, dim))
    }

    fn max_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::max_axis(x, axis))
    }

    fn min_axis(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::min_axis(x, axis))
    }

    fn log_sum_exp(&self, axis: usize, py: Python<'_>) -> PyResult<Self> {
        self.unary_dispatch(py, |x| coeus_autograd::log_sum_exp(x, axis))
    }
}
