pub mod class;
pub mod factory;
pub mod functions;
pub mod ops;

pub use class::{to_py_err, Device, PyTensor, TensorWrapper};

use pyo3::prelude::*;

pub fn register(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyTensor>()?;
    m.add_class::<Device>()?;

    // Submodules registration
    factory::register(py, m)?;
    functions::register(py, m)?;
    ops::register(py, m)?;

    Ok(())
}
