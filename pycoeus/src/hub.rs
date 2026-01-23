use pyo3::prelude::*;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<HubManager>()?;
    m.add_class::<ModelInfo>()?;
    Ok(())
}

/// Hub Manager (placeholder)
#[pyclass(name = "HubManager", module = "_coeus")]
pub struct HubManager;

#[pymethods]
impl HubManager {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Hub manager not yet implemented in Coeus",
        ))
    }
}

/// Model Info (placeholder)
#[pyclass(name = "ModelInfo", module = "_coeus")]
pub struct ModelInfo;

#[pymethods]
impl ModelInfo {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Model info not yet implemented in Coeus",
        ))
    }
}
