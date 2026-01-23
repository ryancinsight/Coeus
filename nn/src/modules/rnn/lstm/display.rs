//! LSTM Display trait implementation.

use backend::Backend;
use dtype::DataType;
use std::fmt;
use storage::{Storage, StorageFromVec, StorageToDense};

use super::core::LSTM;

impl<B, S, T> fmt::Display for LSTM<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T>,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LSTM(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.bidirectional
        )
    }
}
