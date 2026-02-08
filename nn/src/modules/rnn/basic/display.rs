//! Basic RNN Display trait implementation.

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use std::fmt;
use storage::{Storage, StorageFromVec, StorageToDense};

use super::core::RNN;

impl<B, S, T> fmt::Display for RNN<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + std::cmp::PartialOrd,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RNN(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, bidirectional={})",
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.bidirectional
        )
    }
}
