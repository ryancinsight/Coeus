pub mod arithmetic;
pub mod conversion;
pub mod indexing;
pub mod reduction;
pub mod comparison;

pub use crate::tensor::wrapper::{TensorWrapper, WrapTensor};
use pyo3::prelude::*;

pub fn register(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

// Methods moved to class.rs for correct PyO3 exposure
