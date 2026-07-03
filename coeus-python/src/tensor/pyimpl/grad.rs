// ── PyTensor gradient utilities ──

use pyo3::prelude::*;

use super::PyTensor;

#[pymethods]
impl PyTensor {
    #[getter]
    fn grad(&self) -> Option<Vec<f64>> {
        self.inner
            .grad()
            .map(|g| g.to_contiguous().as_slice().to_vec())
    }

    fn backward(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.backward());
        Ok(())
    }

    fn detach(&self) -> Self {
        Self {
            inner: coeus_autograd::Var::new(self.inner.tensor.clone(), false),
        }
    }

    fn requires_grad_(&mut self, requires_grad: bool) -> Self {
        if requires_grad && self.inner.grad.is_none() {
            let t = self.inner.tensor.clone();
            self.inner = coeus_autograd::Var::new(t, true);
        } else if !requires_grad && self.inner.grad.is_some() {
            let t = self.inner.tensor.clone();
            self.inner = coeus_autograd::Var::new(t, false);
        }
        self.clone()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
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
        self.inner.tensor = coeus_tensor::Tensor::from_slice(shape, &data);
        Ok(())
    }
}
