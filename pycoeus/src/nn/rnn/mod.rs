pub mod gru;
pub mod lstm;
#[path = "rnn.rs"]
pub mod rnn_layer;

pub use gru::*;
pub use lstm::*;
pub use rnn_layer::*;

use pyo3::prelude::*;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRNN>()?;
    m.add_class::<PyLSTM>()?;
    m.add_class::<PyGRU>()?;
    m.add_class::<PyGRUCell>()?;

    let dict = m.dict();
    dict.set_item("RNN", m.getattr("RNN")?)?;
    dict.set_item("LSTM", m.getattr("LSTM")?)?;
    dict.set_item("GRU", m.getattr("GRU")?)?;
    dict.set_item("GRUCell", m.getattr("GRUCell")?)?;

    Ok(())
}

pub(crate) fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: RNN error: {}", e))
}
