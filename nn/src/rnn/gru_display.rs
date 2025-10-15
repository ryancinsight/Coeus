//! GRU Display trait implementation.
//!
//! This module provides human-readable display formatting for GRU layers.

use std::fmt;

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::{Storage, StorageFromVec};

use crate::rnn::gru_core::GRU;

impl<B, S, T> fmt::Display for GRU<B, S, T>
where
    B: Backend,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GRU(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.bidirectional
        )
    }
}
