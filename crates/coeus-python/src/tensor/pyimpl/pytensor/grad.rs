//! The `grad` getter.

use super::PyTensor;
use crate::error::map_backend_error;
use pyo3::prelude::*;

#[pymethods]
impl PyTensor {
    #[getter]
    fn grad(&self) -> PyResult<Option<Vec<f64>>> {
        self.inner
            .grad()
            .map(|gradient| {
                gradient
                    .to_contiguous()
                    .map(|tensor| tensor.as_slice().to_vec())
                    .map_err(map_backend_error)
            })
            .transpose()
    }
}
