//! Python bindings for optimizers
//!
//! This module exposes optimizers to Python via PyO3, using a macro-based
//! approach to reduce code duplication.

pub mod adadelta;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod adamw;
pub mod asgd;
pub mod base;
pub mod lr_scheduler;
pub mod nadam;
pub mod radam;
pub mod rmsprop;
pub mod rprop;
pub mod sgd;

use pyo3::prelude::*;

pub use self::adadelta::PyAdadelta;
pub use self::adagrad::PyAdagrad;
pub use self::adam::PyAdam;
pub use self::adamax::PyAdamax;
pub use self::adamw::PyAdamW;
pub use self::lr_scheduler::{
    PyCosineAnnealingLR, PyExponentialLR, PyMultiStepLR, PyOneCycleLR, PyReduceLROnPlateau,
    PyStepLR,
};
pub use self::nadam::PyNAdam;
pub use self::radam::PyRAdam;
pub use self::rmsprop::PyRMSprop;
pub use self::rprop::PyRprop;
pub use self::sgd::PySGD;
pub use self::asgd::PyASGD;

pub use optim::BaseOptimizer;

pub fn register(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyAdagrad>()?;
    m.add_class::<PyRMSprop>()?;
    m.add_class::<PyAdadelta>()?;
    m.add_class::<PyAdamax>()?;
    m.add_class::<PyNAdam>()?;
    m.add_class::<PyRAdam>()?;
    m.add_class::<PyASGD>()?;
    m.add_class::<PyRprop>()?;
    m.add_class::<PyStepLR>()?;
    m.add_class::<PyExponentialLR>()?;
    m.add_class::<PyCosineAnnealingLR>()?;
    m.add_class::<PyMultiStepLR>()?;
    m.add_class::<PyReduceLROnPlateau>()?;
    m.add_class::<PyOneCycleLR>()?;
    Ok(())
}
