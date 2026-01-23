pub mod adaptive;
pub mod avg;
pub mod max;

pub use adaptive::*;
pub use avg::*;
pub use max::*;

use pyo3::prelude::*;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMaxPool1d>()?;
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyMaxPool3d>()?;
    m.add_class::<PyAvgPool1d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyAvgPool3d>()?;
    m.add_class::<PyAdaptiveAvgPool1d>()?;
    m.add_class::<PyAdaptiveAvgPool2d>()?;
    m.add_class::<PyAdaptiveAvgPool3d>()?;
    m.add_class::<PyAdaptiveMaxPool1d>()?;
    m.add_class::<PyAdaptiveMaxPool2d>()?;
    m.add_class::<PyAdaptiveMaxPool3d>()?;

    let dict = m.dict();
    dict.set_item("MaxPool1d", m.getattr("MaxPool1d")?)?;
    dict.set_item("MaxPool2d", m.getattr("MaxPool2d")?)?;
    dict.set_item("MaxPool3d", m.getattr("MaxPool3d")?)?;
    dict.set_item("AvgPool1d", m.getattr("AvgPool1d")?)?;
    dict.set_item("AvgPool2d", m.getattr("AvgPool2d")?)?;
    dict.set_item("AvgPool3d", m.getattr("AvgPool3d")?)?;
    dict.set_item("AdaptiveAvgPool1d", m.getattr("AdaptiveAvgPool1d")?)?;
    dict.set_item("AdaptiveAvgPool2d", m.getattr("AdaptiveAvgPool2d")?)?;
    dict.set_item("AdaptiveAvgPool3d", m.getattr("AdaptiveAvgPool3d")?)?;
    dict.set_item("AdaptiveMaxPool1d", m.getattr("AdaptiveMaxPool1d")?)?;
    dict.set_item("AdaptiveMaxPool2d", m.getattr("AdaptiveMaxPool2d")?)?;
    dict.set_item("AdaptiveMaxPool3d", m.getattr("AdaptiveMaxPool3d")?)?;

    Ok(())
}

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Pooling error: {}", e))
}
