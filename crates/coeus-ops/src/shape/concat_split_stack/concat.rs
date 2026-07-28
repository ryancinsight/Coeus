// ── Concatenation ──
// Concatenates a list of tensors along a given dimension.
// Zero-copy when a single input is given (returns a clone of the storage view).

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar,
};
use coeus_tensor::Tensor;

/// Concatenate `tensors` along `dim`.
///
/// All tensors must have the same shape in every dimension except `dim`.
///
/// # Errors
/// Returns the backend error type for invalid input or materialization failure.
#[inline]
pub fn cat<T: Scalar, B: ComputeBackend + Default>(
    tensors: &[&Tensor<T, B>],
    dim: usize,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if tensors.is_empty() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "cat",
            reason: "input list is empty".to_owned(),
        }));
    }
    // Fast path: single input.
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }

    let backend = B::default();
    let ndim = tensors[0].ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "cat",
            axis: dim,
            rank: ndim,
        }));
    }

    // Validate shapes and compute output dim size.
    let mut out_shape = tensors[0].shape().to_vec();
    let mut out_dim_size = 0usize;
    for tensor in tensors {
        if tensor.ndim() != ndim {
            return Err(B::Error::from(BackendError::LayoutRankMismatch {
                operation: "cat",
                lhs: ndim,
                rhs: tensor.ndim(),
            }));
        }
        for d in 0..ndim {
            if d != dim && tensor.shape()[d] != out_shape[d] {
                return Err(B::Error::from(BackendError::ShapeMismatch {
                    operation: "cat",
                    lhs: out_shape.clone(),
                    rhs: tensor.shape().to_vec(),
                }));
            }
        }
        out_dim_size = out_dim_size
            .checked_add(tensor.shape()[dim])
            .ok_or_else(|| {
                B::Error::from(BackendError::Overflow {
                    operation: "cat",
                    reason: "output dimension",
                })
            })?;
    }
    out_shape[dim] = out_dim_size;

    let layouts: Vec<_> = tensors.iter().map(|tensor| tensor.layout()).collect();
    let inputs: Vec<_> = tensors
        .iter()
        .map(|tensor| tensor.storage().as_slice())
        .collect();
    let values = coeus_leto::concat_values(&layouts, &inputs, dim).map_err(|error| {
        B::Error::from(BackendError::Storage {
            operation: "cat",
            reason: error.to_string(),
        })
    })?;
    Tensor::from_slice_on(out_shape, &values, &backend)
}
