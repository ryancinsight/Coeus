// ── Unary kernel ──
// Generic element-wise unary operation kernel.

use crate::backend_ops::{ElementwiseOps, UnaryOp};
use coeus_core::Scalar;
use coeus_tensor::Tensor;

/// Apply element-wise unary operation to `input`, returning a new tensor.
///
/// Uses `Tensor::alloc_on` (no zero-init) because every output element is
/// unconditionally overwritten by the kernel.
#[inline]
pub fn elementwise_unary<T: Scalar, B: ElementwiseOps<T>>(
    input: &Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<Tensor<T, B>, B::Error> {
    let mut out = Tensor::alloc_on(input.shape_cloned(), backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_unary(op, input.storage(), input.layout(), out_storage, out_layout)?;

    Ok(out)
}

/// Apply element-wise unary operation to `input` in-place.
#[inline]
pub fn elementwise_unary_assign<T: Scalar, B: ElementwiseOps<T>>(
    input: &mut Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<(), B::Error> {
    let (storage, layout) = input.storage_and_layout_mut();
    backend.elementwise_unary_assign(op, storage, layout)
}

/// Apply element-wise unary operation to `input`, writing result to `out`.
#[inline]
pub fn elementwise_unary_to<T: Scalar, B: ElementwiseOps<T>>(
    input: &Tensor<T, B>,
    out: &mut Tensor<T, B>,
    backend: &B,
    op: UnaryOp,
) -> Result<(), B::Error> {
    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.elementwise_unary(op, input.storage(), input.layout(), out_storage, out_layout)
}

#[cfg(test)]
mod tests {
    use super::elementwise_unary_assign;
    use crate::{ElementwiseOps, UnaryOp};
    use coeus_core::{ComputeBackend, CpuAddressableStorage, Layout, SequentialBackend};
    use coeus_tensor::Tensor;

    #[test]
    fn unary_assignment_detaches_shared_storage() {
        let backend = SequentialBackend::new();
        let original = Tensor::from_slice([4], &[-3.0_f32, -1.0, 0.0, 2.0]);
        let mut assigned = original.clone();

        elementwise_unary_assign(&mut assigned, &backend, UnaryOp::Neg).expect("neg assignment");

        assert_eq!(assigned.as_slice(), &[3.0, 1.0, 0.0, -2.0]);
        assert_eq!(original.as_slice(), &[-3.0, -1.0, 0.0, 2.0]);
    }

    #[test]
    fn unary_assignment_preserves_input_on_provider_error() {
        let backend = SequentialBackend::new();
        let mut storage = backend.allocate::<f32>(2);
        backend.copy_to_device(&[-3.0, 2.0], &mut storage);
        let mut invalid_layout = Layout::new([3].into());

        backend
            .elementwise_unary_assign(UnaryOp::Neg, &mut storage, &mut invalid_layout)
            .expect_err("layout exceeding storage must fail");

        assert_eq!(storage.as_slice(), &[-3.0, 2.0]);
        assert_eq!(invalid_layout.shape(), &[3]);
    }
}
