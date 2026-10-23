//! Python wrappers for the recurrent cells and sequence modules.
//!
//! One file per exposed class: `LSTMCell`, `GRUCell`, `RNNCell`,
//! `Bidirectional` and the multi-step `Gru`. Each holds its weights as Python
//! tensors, so `.data` assignment and the optimizers drive them.

mod bidirectional;
mod gru;
mod gru_cell;
mod lstm_cell;
mod rnn_cell;

pub use bidirectional::PyBidirectional;
pub use gru::PyGru;
pub use gru_cell::PyGRUCell;
pub use lstm_cell::PyLSTMCell;
pub use rnn_cell::PyRNNCell;
