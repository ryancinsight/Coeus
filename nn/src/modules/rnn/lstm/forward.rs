//! LSTM forward pass trait.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::Result;

pub trait LSTMForward<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    #[allow(clippy::type_complexity)]
    fn forward_layer_unidirectional_lstm(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        h: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        c: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        weight_idx: usize,
        dims: (usize, usize, usize),
    ) -> Result<(
        Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        (
            Tensor<CpuBackend<T>, DenseStorage<T>, T>,
            Tensor<CpuBackend<T>, DenseStorage<T>, T>,
        ),
    )>;
}
