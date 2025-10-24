//! Recurrent Neural Network (RNN) layers for sequence modeling.
//!
//! This module provides RNN, LSTM, and GRU layers for processing sequential data.
//! All implementations support:
//! - Bidirectional processing
//! - Multi-layer stacking
//! - Batch-first or sequence-first input formats
//! - Hidden state management

pub mod gru_core;
pub mod gru_display;
pub mod gru_forward;
pub mod gru_module;
pub mod gru_tests;
pub mod lstm;
#[allow(clippy::module_inception)]
pub mod rnn;

// Re-export the main types for convenience
pub use gru_core::GRU;
pub use lstm::LSTM;
pub use rnn::RNN;
