use crate::{Result, Tensor, TensorError};
use backend::Backend;
use dtype::DataType;
use num_traits::Float;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the value maxnorm.
pub fn renorm<
    T: DataType + Float,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    p: T,
    dim: usize,
    maxnorm: T,
) -> Result<Tensor<B, S, T>> {
    let shape = tensor.shape().dims();
    if dim >= shape.len() {
        return Err(TensorError::InvalidDimension {
            dim,
            ndim: shape.len(),
        });
    }

    let mut dense = tensor.to_dense_generic()?;
    let data = dense.as_mut_slice();

    let stride: usize = shape.iter().skip(dim + 1).product();
    let outer_size: usize = shape.iter().take(dim).product();
    let dim_size = shape[dim];

    for outer in 0..outer_size {
        for inner in 0..stride {
            let base = outer * dim_size * stride + inner;

            // Calculate p-norm along dim
            let mut norm = T::zero();
            for i in 0..dim_size {
                norm = norm + data[base + i * stride].abs().powf(p);
            }
            norm = norm.powf(T::one() / p);

            if norm > maxnorm {
                let scale = maxnorm / norm;
                for i in 0..dim_size {
                    data[base + i * stride] = data[base + i * stride] * scale;
                }
            }
        }
    }

    Tensor::from_vec(dense.storage().as_slice().to_vec(), shape)
}
