// ── PyTensor in-place mutation operations ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
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
}
