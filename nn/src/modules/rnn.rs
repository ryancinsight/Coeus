//! Recurrent neural network layers
//!
//! This module provides RNN, LSTM, and GRU layers for sequence processing.
//!
//! ## Organization
//!
//! This module has been refactored for better modularity:
//! - `Rnn` and `RnnCell` are in separate rnn_types module
//! - `Lstm` and `LstmCell` are in separate lstm module
//! - `Gru` and `GruCell` are in separate gru module
//!
//! ## Mathematical Foundation
//!
//! ### RNN Forward Pass
//! ```math
//! h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
//! y_t = W_hy * h_t + b_y
//! ```
//!
//! ## References
//!
//! - [Recurrent Neural Networks Tutorial](https://www.deeplearningbook.org/contents/rnn.html)
//! - [PyTorch RNN Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

// Re-export types from specialized modules
pub use crate::modules::rnn_types::{Rnn, RnnCell};
pub use crate::modules::lstm::{Lstm, LstmCell};
pub use crate::modules::gru::{Gru, GruCell};

// Type alias for LSTM forward pass return value to reduce type complexity
pub use crate::modules::lstm::LstmOutput;



