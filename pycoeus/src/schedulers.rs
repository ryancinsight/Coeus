use pyo3::prelude::*;
use pyo3::pyclass;

/// CosineAnnealingWarmRestarts scheduler (placeholder)
#[pyclass(name = "CosineAnnealingWarmRestarts", module = "_coeus")]
pub struct CosineAnnealingWarmRestarts;

#[pymethods]
impl CosineAnnealingWarmRestarts {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CosineAnnealingWarmRestarts scheduler not yet implemented in Coeus"
        ))
    }
}
