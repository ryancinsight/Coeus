//! Standard deviation reduction operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Computes the standard deviation of elements along specified dimensions.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `dims` - Dimensions to reduce over. None means all dimensions.
/// * `keepdim` - Whether to keep the reduced dimensions
/// * `correction` - Degrees of freedom correction (0 for population std, 1 for sample std)
pub fn std<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: Option<&[usize]>,
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + Float + 'static + FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T>,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
{
    // Compute variance
    let var_t = super::var(tensor, dims, keepdim, correction)?;
    
    // Take square root
    let data: Vec<T> = var_t.as_slice().iter().map(|&x| x.sqrt()).collect();
    let result = Tensor::from_vec_with_backend(data, var_t.shape().dims(), var_t.backend.clone())?;

    Ok(result)
}
