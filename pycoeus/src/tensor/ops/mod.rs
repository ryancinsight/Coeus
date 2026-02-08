pub mod arithmetic;
pub mod conversion;
pub mod indexing;
pub mod reduction;
pub mod comparison;
pub mod activation;
pub mod inplace;
pub mod linalg;
pub mod math;
pub mod shape;

pub use crate::tensor::wrapper::{TensorWrapper, WrapTensor};
use pyo3::prelude::*;

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    arithmetic::register(py, m)?;
    conversion::register(py, m)?;
    indexing::register(py, m)?;
    reduction::register(py, m)?;
    comparison::register(py, m)?;
    activation::register(py, m)?;
    inplace::register(py, m)?;
    linalg::register(py, m)?;
    math::register(py, m)?;
    shape::register(py, m)?;
    Ok(())
}

// Methods moved to class.rs for correct PyO3 exposure
