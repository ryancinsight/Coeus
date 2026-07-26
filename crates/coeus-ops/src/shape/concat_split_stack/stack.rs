// ── Stacking ──
// Stacks equal-shaped tensors along a new dimension.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar, Shape};
use coeus_tensor::Tensor;

/// Stack `tensors` along a new dimension `dim`.
///
/// All tensors must have identical shape. `dim` may be any axis in
/// `0..=tensors[0].ndim()`.
///
/// # Panics
/// - `tensors` is empty.
/// - Any tensor shape differs from the first tensor.
/// - `dim` is outside `0..=ndim`.
#[inline]
pub fn stack<T: Scalar, B: ComputeBackend + Default>(
    tensors: &[&Tensor<T, B>],
    dim: usize,
) -> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert!(!tensors.is_empty(), "stack: input list is empty");

    let backend = B::default();
    let ndim = tensors[0].ndim();
    assert!(
        dim <= ndim,
        "stack: dim {dim} out of range for insertion into {ndim}D tensor"
    );

    let base_shape = tensors[0].shape();
    for tensor in tensors {
        assert_eq!(
            tensor.shape(),
            base_shape,
            "stack: all tensors must have identical shape"
        );
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
    let values = coeus_leto::stack_values(&layouts, &inputs, dim).expect("coeus-leto stack failed");
    Tensor::from_slice_on(out_shape, &values, &backend)
}
