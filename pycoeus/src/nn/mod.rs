//! Python bindings for the nn crate.
//!
//! This module exposes Neural Network components to Python via PyO3.

use pyo3::prelude::*;

pub mod activations;
pub mod bilinear;
pub mod common;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod linear;
pub mod loss;
pub mod normalization;
pub mod pooling;
pub mod rnn;
pub mod sequential;
pub mod utility;
pub mod distance;

/// Register the nn module functionality with the Python module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    linear::register(py, m)?;
    bilinear::register(py, m)?;
    activations::register(py, m)?;
    common::register(py, m)?;
    conv::register(py, m)?;
    normalization::register(py, m)?;
    sequential::register(py, m)?;
    dropout::register(py, m)?;
    embedding::register(py, m)?;
    rnn::register(py, m)?;
    pooling::register(py, m)?;
    loss::register(py, m)?;
    distance::register(py, m)?;
    utility::register(py, m)?;
    Ok(())
}
