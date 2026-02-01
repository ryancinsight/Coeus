use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec};

/// In-place clamp
pub fn clamp_<
    T: DataType + PartialOrd + Clone,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    tensor: &mut Tensor<B, S, T>,
    min: T,
    max: T,
) -> Result<()> {
    // Note: We need a way to mutate the storage.
    // Assuming tensor.as_mut_slice() exists or similar.
    // For now, let's use a non-optimal way if mutate is not exposed.
    let dims = tensor.shape().dims().to_vec();
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|x| {
            if x < &min {
                min.clone()
            } else if x > &max {
                max.clone()
            } else {
                x.clone()
            }
        })
        .collect();
    *tensor = Tensor::from_vec_with_backend(data, &dims, tensor.backend.clone())?;
    Ok(())
}

pub fn clamp_min_<
    T: DataType + PartialOrd + Clone,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    tensor: &mut Tensor<B, S, T>,
    min: T,
) -> Result<()> {
    let dims = tensor.shape().dims().to_vec();
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|x| if x < &min { min.clone() } else { x.clone() })
        .collect();
    *tensor = Tensor::from_vec_with_backend(data, &dims, tensor.backend.clone())?;
    Ok(())
}

pub fn clamp_max_<
    T: DataType + PartialOrd + Clone,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
>(
    tensor: &mut Tensor<B, S, T>,
    max: T,
) -> Result<()> {
    let dims = tensor.shape().dims().to_vec();
    let data: Vec<T> = tensor
        .as_slice()
        .iter()
        .map(|x| if x > &max { max.clone() } else { x.clone() })
        .collect();
    *tensor = Tensor::from_vec_with_backend(data, &dims, tensor.backend.clone())?;
    Ok(())
}
