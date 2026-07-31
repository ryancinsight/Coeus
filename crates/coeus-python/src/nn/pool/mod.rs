//! Python pooling layer bindings.

mod adaptive;
mod avg;
mod global;
mod max;

pub use adaptive::{
    PyAdaptiveAvgPool1d, PyAdaptiveAvgPool2d, PyAdaptiveMaxPool1d, PyAdaptiveMaxPool2d,
};
pub use avg::{PyAvgPool1d, PyAvgPool2d, PyAvgPool3d};
pub use global::{
    PyGlobalAvgPool1d, PyGlobalAvgPool2d, PyGlobalAvgPool3d, PyGlobalMaxPool2d, PyGlobalMaxPool3d,
};
pub use max::{PyMaxPool1d, PyMaxPool2d, PyMaxPool3d};
