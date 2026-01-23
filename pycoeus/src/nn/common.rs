use crate::tensor::PyTensor;
use pyo3::prelude::*;

// ============ Identity ============
#[pyclass(name = "Identity", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyIdentity;

#[pymethods]
impl PyIdentity {
    #[new]
    fn new() -> Self {
        PyIdentity
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(input.clone())
    }
}

// ============ Flatten ============
#[pyclass(name = "Flatten", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyFlatten {
    pub start_dim: isize,
    pub end_dim: isize,
}

#[pymethods]
impl PyFlatten {
    #[new]
    #[pyo3(signature = (start_dim=1, end_dim=-1))]
    fn new(start_dim: isize, end_dim: isize) -> Self {
        PyFlatten { start_dim, end_dim }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        input.flatten(self.start_dim as usize, self.end_dim)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdentity>()?;
    m.add_class::<PyFlatten>()?;

    let dict = m.dict();
    dict.set_item("Identity", m.getattr("Identity")?)?;
    dict.set_item("Flatten", m.getattr("Flatten")?)?;

    Ok(())
}
