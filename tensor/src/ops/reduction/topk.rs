use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::Storage;

/// Returns the k largest elements of the given input tensor along a given dimension.
///
/// If `dim` is not given, the last dimension of the input is chosen.
///
/// # Arguments
/// * `tensor` - The input tensor.
/// * `k` - The number of top elements to return.
/// * `dim` - The dimension to sort along.
/// * `largest` - If true, return largest elements, otherwise smallest.
/// * `sorted` - If true, returned elements are sorted.
///
/// # Returns
/// A tuple of (values, indices) where values and indices are the top k elements.
pub fn topk<B, S, T>(
    _tensor: &Tensor<B, S, T>,
    _k: usize,
    _dim: usize,
    _largest: bool,
    _sorted: bool,
) -> Result<(Tensor<B, S, T>, Tensor<B, S, T>)>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone,
    T: DataType + PartialOrd,
{
    // Stubbed until Backend supports i64 indices return or different backend instance is provided
    Err(crate::TensorError::NotImplemented("topk not fully implemented yet due to backend limitations".into()))
}
