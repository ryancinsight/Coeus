//! Gated activation kernels.

use super::super::kernel::elementwise_unary;
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;

/// Gated Linear Unit (GLU): splits `input` in half along `dim`, returns
/// `first_half * sigmoid(second_half)`.
///
/// `input.shape()[dim]` must be even.
/// Equivalent to `torch.nn.functional.glu(input, dim)`.
#[inline]
pub fn glu<T: Float, B: BackendOps<T> + Default>(
    input: &Tensor<T, B>,
    dim: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let ndim = input.ndim();
    if dim >= ndim {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "glu",
            axis: dim,
            rank: ndim,
        }));
    }
    let dim_size = input.shape()[dim];
    if !dim_size.is_multiple_of(2) {
        return Err(B::Error::from(BackendError::Storage {
            operation: "glu",
            reason: format!("axis {dim} size {dim_size} must be even"),
        }));
    }
    let half = dim_size / 2;
    let parts = crate::shape::split(input, half, dim)?;
    let [a_part, b_part] = parts.as_slice() else {
        return Err(B::Error::from(BackendError::Storage {
            operation: "glu",
            reason: "split did not produce two halves".to_owned(),
        }));
    };
    let gate = elementwise_unary(b_part, backend, UnaryOp::Sigmoid)?;
    Ok(crate::binary::mul(a_part, &gate, backend)?)
}
