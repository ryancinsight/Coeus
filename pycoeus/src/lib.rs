use pyo3::prelude::*;
use pyo3::PyResult;

mod nn;
mod tensor;

use nn::{Adam, Conv2d, CrossEntropyLoss, Gru, Linear, Lstm, MseLoss, NNModule, ReLU, Rnn, Sgd};
use tensor::{Device, PyTensor};

/// Python bindings for Coeus tensor library
/// Provides PyTorch-compatible API with automatic differentiation
#[pymodule]
fn pycoeus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tensor operations
    m.add_class::<PyTensor>()?;
    m.add_class::<Device>()?;

    // Neural network modules
    m.add_class::<NNModule>()?;
    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<ReLU>()?;
    m.add_class::<Rnn>()?;
    m.add_class::<Lstm>()?;
    m.add_class::<Gru>()?;

    // Loss functions
    m.add_class::<MseLoss>()?;
    m.add_class::<CrossEntropyLoss>()?;

    // Optimizers
    m.add_class::<Sgd>()?;
    m.add_class::<Adam>()?;

    Ok(())
}
