use crate::tensor::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::error::Error;

/// Convert a neural-network initialization failure to its Python exception class.
pub(crate) fn map_initialization_error<E>(error: coeus_nn::init::InitializationError<E>) -> PyErr
where
    E: Error + 'static,
{
    match error {
        coeus_nn::init::InitializationError::Backend { operation, source } => {
            PyRuntimeError::new_err(format!(
                "{operation} backend initialization failed: {source}"
            ))
        }
        domain => PyValueError::new_err(domain.to_string()),
    }
}

fn extract_fan(name: &str, fan: &Bound<'_, PyAny>) -> PyResult<usize> {
    let fan = fan.extract::<i128>().map_err(|_| {
        PyValueError::new_err(format!("{name} must be an integer representable as usize"))
    })?;
    usize::try_from(fan).map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an integer representable as usize, got {fan}"
        ))
    })
}

/// Fill `tensor` in-place with values drawn uniformly from `[a, b)`.
#[pyfunction]
pub fn uniform_(tensor: &mut PyTensor, a: f64, b: f64) -> PyResult<()> {
    coeus_nn::init::uniform(&mut tensor.inner, a, b).map_err(map_initialization_error)
}

/// Fill `tensor` in-place with values drawn from N(`mean`, `std_dev`²).
#[pyfunction]
pub fn normal_(tensor: &mut PyTensor, mean: f64, std_dev: f64) -> PyResult<()> {
    coeus_nn::init::normal(&mut tensor.inner, mean, std_dev).map_err(map_initialization_error)
}

/// Fill `tensor` in-place with the constant `val`.
#[pyfunction]
pub fn constant_(tensor: &mut PyTensor, val: f64) {
    coeus_nn::init::constant(&mut tensor.inner, val);
}

/// Fill `tensor` in-place with zeros.
#[pyfunction]
pub fn zeros_(tensor: &mut PyTensor) {
    coeus_nn::init::zeros(&mut tensor.inner);
}

/// Fill `tensor` in-place with ones.
#[pyfunction]
pub fn ones_(tensor: &mut PyTensor) {
    coeus_nn::init::ones(&mut tensor.inner);
}

/// Apply Xavier uniform initialization (gain=1) in-place.
#[pyfunction]
pub fn xavier_uniform_(
    tensor: &mut PyTensor,
    fan_in: &Bound<'_, PyAny>,
    fan_out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let fan_in = extract_fan("fan_in", fan_in)?;
    let fan_out = extract_fan("fan_out", fan_out)?;
    coeus_nn::init::xavier_uniform(&mut tensor.inner, fan_in, fan_out)
        .map_err(map_initialization_error)
}

/// Apply Xavier normal initialization (gain=1) in-place.
#[pyfunction]
pub fn xavier_normal_(
    tensor: &mut PyTensor,
    fan_in: &Bound<'_, PyAny>,
    fan_out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let fan_in = extract_fan("fan_in", fan_in)?;
    let fan_out = extract_fan("fan_out", fan_out)?;
    coeus_nn::init::xavier_normal(&mut tensor.inner, fan_in, fan_out)
        .map_err(map_initialization_error)
}

/// Apply Kaiming uniform initialization (mode=fan_in, nonlinearity=relu) in-place.
#[pyfunction]
pub fn kaiming_uniform_(tensor: &mut PyTensor, fan_in: &Bound<'_, PyAny>) -> PyResult<()> {
    let fan_in = extract_fan("fan_in", fan_in)?;
    coeus_nn::init::kaiming_uniform(&mut tensor.inner, fan_in).map_err(map_initialization_error)
}

/// Apply Kaiming normal initialization (mode=fan_in, nonlinearity=relu) in-place.
#[pyfunction]
pub fn kaiming_normal_(tensor: &mut PyTensor, fan_in: &Bound<'_, PyAny>) -> PyResult<()> {
    let fan_in = extract_fan("fan_in", fan_in)?;
    coeus_nn::init::kaiming_normal(&mut tensor.inner, fan_in).map_err(map_initialization_error)
}
