use pyo3::prelude::*;
use pyo3::pyclass;

/// FFT operation (placeholder)
#[pyclass(name = "FFT", module = "_coeus")]
pub struct FFT;

#[pymethods]
impl FFT {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "FFT not yet implemented in Coeus"
        ))
    }
}

/// IFFT operation (placeholder)
#[pyclass(name = "IFFT", module = "_coeus")]
pub struct IFFT;

#[pymethods]
impl IFFT {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "IFFT not yet implemented in Coeus"
        ))
    }
}
