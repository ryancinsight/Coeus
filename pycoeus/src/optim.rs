use pyo3::prelude::*;
use pyo3::pyclass;
use coeus_optim::{Adam as RustAdam, SGD as RustSGD};
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

/// Adam optimizer
#[pyclass(name = "Adam", module = "_coeus", unsendable)]
pub struct Adam {
    pub inner: RustAdam<CpuBackend, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (lr, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> PyResult<Self> {
        let adam = RustAdam::new(lr, beta1, beta2, epsilon).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create Adam optimizer: {:?}", e))
        })?;
        Ok(Adam { inner: adam })
    }

    fn step(&mut self) -> PyResult<()> {
        // Implementation will be added when optimizers support step()
        Ok(())
    }
}

/// SGD optimizer
#[pyclass(name = "SGD", module = "_coeus", unsendable)]
pub struct Sgd {
    pub inner: RustSGD<CpuBackend, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl Sgd {
    #[new]
    #[pyo3(signature = (lr, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=false))]
    fn new(lr: f64, momentum: f64, weight_decay: f64, dampening: f64, nesterov: bool) -> PyResult<Self> {
        let sgd = RustSGD::new(lr, momentum, weight_decay, dampening, nesterov).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create SGD optimizer: {:?}", e))
        })?;
        Ok(Sgd { inner: sgd })
    }

    fn step(&mut self) -> PyResult<()> {
        // Implementation will be added when optimizers support step()
        Ok(())
    }
}

/// AdamW optimizer (placeholder - not implemented in core)
#[pyclass(name = "AdamW", module = "_coeus")]
pub struct AdamW;

#[pymethods]
impl AdamW {
    #[new]
    fn new(_lr: f64) -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "AdamW optimizer not yet implemented in Coeus core"
        ))
    }
}

/// Adagrad optimizer (placeholder - not implemented in core)
#[pyclass(name = "Adagrad", module = "_coeus")]
pub struct Adagrad;

#[pymethods]
impl Adagrad {
    #[new]
    fn new(_lr: f64) -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Adagrad optimizer not yet implemented in Coeus core"
        ))
    }
}
