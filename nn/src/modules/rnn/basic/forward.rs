//! Basic RNN forward pass trait.

use super::core::{CpuTensor, TensorPair};
use crate::core::error::Result;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec};

pub trait RNNForward<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward_layer(
        &self,
        input: &CpuTensor<T>,
        hidden: &CpuTensor<T>,
        weight_idx: usize,
        seq_len: usize,
        batch_size: usize,
        input_size: usize,
    ) -> Result<TensorPair<T>>;
}
