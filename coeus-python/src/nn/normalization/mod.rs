pub mod batchnorm;
pub mod groupnorm;
pub mod instancenorm;
pub mod layernorm;
pub mod rmsnorm;

pub use batchnorm::{PyBatchNorm1d, PyBatchNorm2d, PyBatchNorm3d};
pub use groupnorm::PyGroupNorm;
pub use instancenorm::{PyInstanceNorm1d, PyInstanceNorm2d, PyInstanceNorm3d};
pub use layernorm::PyLayerNorm;
pub use rmsnorm::PyRMSNorm;

use pyo3::prelude::*;

/// Register all normalization classes into the given PyO3 module.
pub fn register_normalization(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyRMSNorm>()?;
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyBatchNorm3d>()?;
    m.add_class::<PyGroupNorm>()?;
    m.add_class::<PyInstanceNorm1d>()?;
    m.add_class::<PyInstanceNorm2d>()?;
    m.add_class::<PyInstanceNorm3d>()?;
    Ok(())
}
