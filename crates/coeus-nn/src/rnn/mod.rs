// ── RNN module ──

/// Bidirectional recurrent wrapper.
pub mod bidirectional;
/// Gated Recurrent Unit cell and sequence module.
pub mod gru;
/// Long Short-Term Memory cell and sequence module.
pub mod lstm;
mod validation;
/// Vanilla (Elman) RNN cell and sequence module.
pub mod vanilla;

pub use bidirectional::Bidirectional;
pub use gru::{GRUCell, Gru};
pub use lstm::{LSTMCell, Lstm};
pub use vanilla::{RNNCell, Rnn, RnnNonlinearity};

use coeus_autograd::Var;

/// A pair of tracked variables on the same backend (e.g. hidden and cell state).
pub(crate) type VarPair<T, B> = (Var<T, B>, Var<T, B>);
