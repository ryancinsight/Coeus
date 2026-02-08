pub mod flatten;
pub mod identity;

pub use flatten::PyFlatten;
pub use identity::PyIdentity;

use pyo3::prelude::*;

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Utility error: {}", e))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlatten>()?;
    m.add_class::<PyIdentity>()?;
    
    let dict = m.dict();
    dict.set_item("Flatten", m.getattr("Flatten")?)?;
    dict.set_item("Identity", m.getattr("Identity")?)?;
    
    Ok(())
}
