//! Gated activation kernels.
#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]

use super::super::kernel::elementwise_unary;
use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float};
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
    assert!(
        dim < ndim,
        "glu: dim {dim} out of bounds for {ndim}D tensor"
    );
    let dim_size = input.shape()[dim];
    assert!(
        dim_size.is_multiple_of(2),
        "glu: dim {dim} size {dim_size} must be even"
    );
    let half = dim_size / 2;
    let mut parts = crate::shape::split(input, half, dim);
    assert_eq!(parts.len(), 2);
    let b_part = parts.pop().unwrap();
    let a_part = parts.pop().unwrap();
    let gate = elementwise_unary(&b_part, backend, UnaryOp::Sigmoid)?;
    Ok(crate::binary::mul(&a_part, &gate, backend))
}
