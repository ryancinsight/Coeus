// ── Split ──
// Splits a tensor into multiple tensors along a given dimension.

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar,
};
use coeus_tensor::Tensor;

/// Split `x` into chunks of size `chunk_size` along `dim`.
///
/// The last chunk may be smaller if `x.shape()[dim]` is not divisible.
///
/// # Errors
/// Returns the backend error type for invalid input or materialization failure.
#[inline]
pub fn split<T: Scalar, B: ComputeBackend + Default>(
    x: &Tensor<T, B>,
    chunk_size: usize,
    dim: usize,
) -> Result<Vec<Tensor<T, B>>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if chunk_size == 0 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "split",
            reason: "chunk_size must be greater than zero".to_owned(),
        }));
    }
    let ndim = x.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "split",
            axis: dim,
            rank: ndim,
        }));
    }

    let backend = B::default();
    let dim_size = x.shape()[dim];
    let sizes: Vec<_> = (0..dim_size)
        .step_by(chunk_size)
        .map(|start| (dim_size - start).min(chunk_size))
        .collect();
    let values = coeus_leto::split_values(x.layout(), x.storage().as_slice(), dim, &sizes)
        .map_err(|error| {
            B::Error::from(BackendError::Storage {
                operation: "split",
                reason: error.to_string(),
            })
        })?;

    values
        .iter()
        .zip(sizes)
        .map(|(chunk, chunk_dim)| {
            let mut out_shape = x.shape().to_vec();
            out_shape[dim] = chunk_dim;
            Tensor::from_slice_on(out_shape, chunk, &backend)
        })
        .map(|result| result)
        .collect()
}
