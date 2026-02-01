use crate::{Result, Tensor, TensorError};
use backend::{Backend, CpuBackend};
use dtype::int::I64;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};

/// Returns the indices of the maximum value of all elements in the input tensor along a given dimension.
pub fn argmax<
    T: DataType + PartialOrd,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<CpuBackend<I64>, DenseStorage<I64>, I64>> {
    let shape = tensor.storage().shape();
    let dims = shape.dims();
    if dim >= dims.len() {
        return Err(TensorError::InvalidDimension {
            dim,
            ndim: dims.len(),
        });
    }

    let dense = tensor.to_dense_generic()?;
    let data = dense.storage().as_slice();

    let stride: usize = dims.iter().skip(dim + 1).product();
    let outer_size: usize = dims.iter().take(dim).product();
    let dim_size = dims[dim];

    let mut res_data = Vec::with_capacity(outer_size * stride);

    for outer in 0..outer_size {
        for inner in 0..stride {
            let base = outer * dim_size * stride + inner;
            let mut max_val = data[base];
            let mut max_idx = 0;
            for i in 1..dim_size {
                let val = data[base + i * stride];
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            res_data.push(I64::new(max_idx as i64));
        }
    }

    let mut res_dims = dims.to_vec();
    if keepdim {
        res_dims[dim] = 1;
    } else {
        res_dims.remove(dim);
    }

    Tensor::from_vec_with_backend(res_data, &res_dims, CpuBackend::default())
}

/// Returns the indices of the minimum value of all elements in the input tensor along a given dimension.
pub fn argmin<
    T: DataType + PartialOrd,
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + crate::ops::TensorStorageOps<T> + 'static,
>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<CpuBackend<I64>, DenseStorage<I64>, I64>> {
    let shape = tensor.storage().shape();
    let dims = shape.dims();
    if dim >= dims.len() {
        return Err(TensorError::InvalidDimension {
            dim,
            ndim: dims.len(),
        });
    }

    let dense = tensor.to_dense_generic()?;
    let data = dense.storage().as_slice();

    let stride: usize = dims.iter().skip(dim + 1).product();
    let outer_size: usize = dims.iter().take(dim).product();
    let dim_size = dims[dim];

    let mut res_data = Vec::with_capacity(outer_size * stride);

    for outer in 0..outer_size {
        for inner in 0..stride {
            let base = outer * dim_size * stride + inner;
            let mut min_val = data[base];
            let mut min_idx = 0;
            for i in 1..dim_size {
                let val = data[base + i * stride];
                if val < min_val {
                    min_val = val;
                    min_idx = i;
                }
            }
            res_data.push(I64::new(min_idx as i64));
        }
    }

    let mut res_dims = dims.to_vec();
    if keepdim {
        res_dims[dim] = 1;
    } else {
        res_dims.remove(dim);
    }

    Tensor::from_vec_with_backend(res_data, &res_dims, CpuBackend::default())
}
