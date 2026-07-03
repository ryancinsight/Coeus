// ── PyTensor comparison operations ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
    fn eq(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if (a - b).abs() < f64::EPSILON * 8.0 {
                    1.0
                } else {
                    0.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }

    fn lt(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if a < b {
                    1.0
                } else {
                    0.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }

    fn gt(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if a > b {
                    1.0
                } else {
                    0.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }

    fn ne(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if (a - b).abs() < f64::EPSILON * 8.0 {
                    0.0
                } else {
                    1.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }

    fn ge(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if a >= b {
                    1.0
                } else {
                    0.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }

    fn le(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let inner_t = py.allow_threads(|| {
            crate::ops::tensor_cmp(&self.inner.tensor, &other.inner.tensor, |a, b| {
                if a <= b {
                    1.0
                } else {
                    0.0
                }
            })
        })?;
        Ok(Self {
            inner: coeus_autograd::Var::new(inner_t, false),
        })
    }
}
