// ── Python tensor iterator ──

use pyo3::prelude::*;

use super::PyTensor;

/// Python iterator over the first dimension of a `PyTensor`.
#[pyclass(name = "TensorIterator")]
pub struct PyTensorIterator {
    pub tensor: PyTensor,
    pub current: usize,
    pub length: usize,
}

#[pymethods]
impl PyTensorIterator {
    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<PyTensor> {
        if self.current >= self.length {
            return None;
        }
        let idx = self.current;
        self.current += 1;
        let ranges: Vec<(usize, usize)> = self
            .tensor
            .inner
            .tensor
            .shape()
            .iter()
            .enumerate()
            .map(|(d, &s)| if d == 0 { (idx, idx + 1) } else { (0, s) })
            .collect();
        let inner = py.allow_threads(|| {
            let sliced = coeus_autograd::slice(&self.tensor.inner, &ranges);
            coeus_autograd::squeeze(&sliced, Some(0))
        });
        Some(PyTensor::from_var(inner))
    }
}
