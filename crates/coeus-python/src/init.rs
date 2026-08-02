use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

fn validate_random_tensor(tensor: &PyTensor) -> PyResult<()> {
    let rank = tensor.inner.tensor.ndim();
    if (1..=coeus_nn::init::MAX_INITIALIZER_RANK).contains(&rank) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "random initialization requires tensor rank in 1..={}, got {rank}",
            coeus_nn::init::MAX_INITIALIZER_RANK
        )))
    }
}

fn validate_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

fn validate_fan(name: &str, fan: &Bound<'_, PyAny>) -> PyResult<usize> {
    let fan = fan.extract::<i128>().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an integer representable as a positive usize"
        ))
    })?;
    usize::try_from(fan)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be a positive usize, got {fan}")))
}

fn validate_fan_sum(
    fan_in: &Bound<'_, PyAny>,
    fan_out: &Bound<'_, PyAny>,
) -> PyResult<(usize, usize)> {
    let fan_in = validate_fan("fan_in", fan_in)?;
    let fan_out = validate_fan("fan_out", fan_out)?;
    fan_in
        .checked_add(fan_out)
        .map(|_| (fan_in, fan_out))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "fan_in + fan_out must be positive and representable, got {fan_in} + {fan_out}"
            ))
        })
}

/// Fill `tensor` in-place with values drawn uniformly from `[a, b)`.
#[pyfunction]
pub fn uniform_(tensor: &mut PyTensor, a: f64, b: f64) -> PyResult<()> {
    validate_random_tensor(tensor)?;
    validate_finite("a", a)?;
    validate_finite("b", b)?;
    if a > b {
        return Err(PyValueError::new_err(format!(
            "uniform lower bound must not exceed upper bound, got {a} > {b}"
        )));
    }
    coeus_nn::init::uniform(&mut tensor.inner, a, b);
    Ok(())
}

/// Fill `tensor` in-place with values drawn from N(`mean`, `std_dev`²).
#[pyfunction]
pub fn normal_(tensor: &mut PyTensor, mean: f64, std_dev: f64) -> PyResult<()> {
    validate_random_tensor(tensor)?;
    validate_finite("mean", mean)?;
    validate_finite("std_dev", std_dev)?;
    if std_dev < 0.0 {
        return Err(PyValueError::new_err(format!(
            "std_dev must be non-negative, got {std_dev}"
        )));
    }
    coeus_nn::init::normal(&mut tensor.inner, mean, std_dev);
    Ok(())
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
    validate_random_tensor(tensor)?;
    let (fan_in, fan_out) = validate_fan_sum(fan_in, fan_out)?;
    coeus_nn::init::xavier_uniform(&mut tensor.inner, fan_in, fan_out);
    Ok(())
}

/// Apply Xavier normal initialization (gain=1) in-place.
#[pyfunction]
pub fn xavier_normal_(
    tensor: &mut PyTensor,
    fan_in: &Bound<'_, PyAny>,
    fan_out: &Bound<'_, PyAny>,
) -> PyResult<()> {
    validate_random_tensor(tensor)?;
    let (fan_in, fan_out) = validate_fan_sum(fan_in, fan_out)?;
    coeus_nn::init::xavier_normal(&mut tensor.inner, fan_in, fan_out);
    Ok(())
}

/// Apply Kaiming uniform initialization (mode=fan_in, nonlinearity=relu) in-place.
#[pyfunction]
pub fn kaiming_uniform_(tensor: &mut PyTensor, fan_in: &Bound<'_, PyAny>) -> PyResult<()> {
    validate_random_tensor(tensor)?;
    let fan_in = validate_fan("fan_in", fan_in)?;
    coeus_nn::init::kaiming_uniform(&mut tensor.inner, fan_in);
    Ok(())
}

/// Apply Kaiming normal initialization (mode=fan_in, nonlinearity=relu) in-place.
#[pyfunction]
pub fn kaiming_normal_(tensor: &mut PyTensor, fan_in: &Bound<'_, PyAny>) -> PyResult<()> {
    validate_random_tensor(tensor)?;
    let fan_in = validate_fan("fan_in", fan_in)?;
    coeus_nn::init::kaiming_normal(&mut tensor.inner, fan_in);
    Ok(())
}
