// ── Unary kernel ──
// Generic element-wise unary operation kernel.

use crate::backend_ops::{BackendOps, UnaryOp};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Apply element-wise unary operation to `input`, returning a new tensor.
///
/// Uses `Tensor::alloc_on` (no zero-init) because every output element is
/// unconditionally overwritten by the kernel.
#[inline]
pub fn elementwise_unary<T: Scalar, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<Tensor<T, B>, B::Error> {
    let mut out = Tensor::alloc_on(input.shape_cloned(), backend)?;

    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.elementwise_unary(op, input.storage(), input.layout(), out_storage, out_layout)?;

    Ok(out)
}

/// Apply element-wise unary operation to `input` in-place.
#[inline]
pub fn elementwise_unary_assign<T: Scalar, B: BackendOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<(), B::Error> {
    let (c, layout) = input.storage_mut_and_layout()?;
    // SAFETY: We cast the mutable reference `c` to an immutable reference `a`
    // to pass as the source buffer. This is safe because:
    // 1. `c` has been made unique (Arc count is 1) via `storage_mut()`.
    // 2. The backend supports in-place / overlapping reads and writes to the same device buffer.
    // 3. We avoid cloning the device buffer (Arc clone), preventing copy-on-write reallocation.
    let a: &B::DeviceBuffer<T> = unsafe { &*(c as *const B::DeviceBuffer<T>) };
    backend.elementwise_unary(op, a, layout, c, layout)
}

/// Apply element-wise unary operation to `input`, writing result to `out`.
#[inline]
pub fn elementwise_unary_to<T: Scalar, B: BackendOps<T>>(
    input: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<(), B::Error> {
    let (out_storage, out_layout) = out.storage_mut_and_layout()?;
    backend.elementwise_unary(op, input.storage(), input.layout(), out_storage, out_layout)
}
