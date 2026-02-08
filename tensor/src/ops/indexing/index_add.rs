//! index_add operation.

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{DataType, I64};
use storage::{Storage, StorageFromVec, StorageToDense};
use num_traits::{Num, FromPrimitive};

/// Accumulate the elements of alpha * source into input by adding to the indices in the order given in index.
///
/// For example, if dim == 0, index[i] == j, then the ith row of source is added to the jth row of input.
pub fn index_add<B, S, T, B2, S2>(
    input: &mut Tensor<B, S, T>,
    dim: usize,
    index: &Tensor<B2, S2, I64>,
    source: &Tensor<B, S, T>,
    alpha: T,
) -> Result<()>
where
    B: Backend<Data = T> + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Num + FromPrimitive + Copy + 'static,
    B2: Backend<Data = I64> + Clone + 'static,
    S2: Storage<I64> + StorageToDense<I64> + StorageFromVec<I64> + 'static,
{
    let rank = input.storage.shape().ndim();
    if dim >= rank {
        return Err(crate::TensorError::InvalidDimension { dim, ndim: rank });
    }

    let index_dense = index.to_dense_generic()?;
    let index_data = index_dense.storage.as_slice();
    
    let input_dims = input.storage.shape().dims();
    let source_dims = source.storage.shape().dims();
    
    if input_dims.len() != source_dims.len() {
        return Err(crate::TensorError::ShapeMismatch {
            expected: input_dims.to_vec(),
            actual: source_dims.to_vec(),
            operation: "index_add",
        });
    }
    
    for i in 0..input_dims.len() {
        if i != dim && input_dims[i] != source_dims[i] {
             return Err(crate::TensorError::ShapeMismatch {
                expected: input_dims.to_vec(),
                actual: source_dims.to_vec(),
                operation: "index_add",
            });
        }
    }
    
    if index_data.len() != source_dims[dim] {
        return Err(crate::TensorError::InvalidInput {
            message: format!("index_add: index length {} must match source.size(dim) {}", index_data.len(), source_dims[dim]),
        });
    }

    let mut input_dense = input.to_dense_generic()?;
    let source_dense = source.to_dense_generic()?;
    
    let input_data = input_dense.storage.as_mut_slice();
    let source_data = source_dense.storage.as_slice();
    
    let stride_input = input.storage.strides()[dim];
    let stride_source = source.storage.strides()[dim];
    let slice_size = stride_input; 
    
    for (i, &idx) in index_data.iter().enumerate() {
        let target_idx = idx.0 as usize;
        if target_idx >= input_dims[dim] {
            return Err(crate::TensorError::InvalidInput {
                message: format!("index_add: index {target_idx} out of bounds for dim {dim} with size {}", input_dims[dim]),
            });
        }
        
        let src_offset = i * stride_source;
        let dst_offset = target_idx * stride_input;
        
        for k in 0..slice_size {
            input_data[dst_offset + k] = input_data[dst_offset + k] + alpha * source_data[src_offset + k];
        }
    }
    
    // Update input storage
    let final_data = input_dense.storage.as_slice().to_vec();
    let new_storage = S::from_vec(final_data, input_dims)?;
    input.storage = new_storage;

    Ok(())
}
