// ── Stacking ──
// Stacks equal-shaped tensors along a new dimension.

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar, Shape,
};
use coeus_tensor::Tensor;

/// Stack `tensors` along a new dimension `dim`.
///
/// All tensors must have identical shape. `dim` may be any axis in
/// `0..=tensors[0].ndim()`.
///
/// # Errors
/// Returns the backend error type for invalid input or materialization failure.
#[inline]
pub fn stack<T: Scalar, B: ComputeBackend + Default>(
    tensors: &[&Tensor<T, B>],
    dim: usize,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    if tensors.is_empty() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "stack",
            reason: "input list is empty".to_owned(),
        }));
    }

    let backend = B::default();
    let ndim = tensors[0].ndim();
    if dim > ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "stack",
            axis: dim,
            rank: ndim + 1,
        }));
    }

    let base_shape = tensors[0].shape();
    for tensor in tensors {
        if tensor.shape() != base_shape {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "stack",
                lhs: base_shape.to_vec(),
                rhs: tensor.shape().to_vec(),
            }));
        }
    }

    let mut out_shape = Shape::with_capacity(ndim + 1);
    for axis in 0..dim {
        out_shape.push(base_shape[axis]);
    }
    out_shape.push(tensors.len());
    for axis in dim..ndim {
        out_shape.push(base_shape[axis]);
    }

    let layouts: Vec<_> = tensors.iter().map(|tensor| tensor.layout()).collect();
    let inputs: Vec<_> = tensors
        .iter()
        .map(|tensor| tensor.storage().as_slice())
        .collect();
    let values = coeus_leto::stack_values(&layouts, &inputs, dim).map_err(|error| {
        B::Error::from(BackendError::Storage {
            operation: "stack",
            reason: error.to_string(),
        })
    })?;
    Tensor::from_slice_on(out_shape, &values, &backend)
}
