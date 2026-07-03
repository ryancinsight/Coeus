// ── PyTensor reduction operations ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
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
}
