//! Provider-owned product reductions.

use crate::backend_ops::{BackendOps, ReductionOp};
use coeus_core::{BackendError, Scalar};
use coeus_tensor::Tensor;

/// Product along a specific axis, reducing it to size 1.
///
/// The reduction uses the multiplicative identity for empty axes, matching the
/// CPU Leto contract and the provider kernels.
#[inline]
pub fn prod_axis<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    axis: usize,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    if axis >= a.ndim() {
        return Err(B::Error::from(BackendError::AxisOutOfRange {
            operation: "prod_axis",
            axis,
            rank: a.ndim(),
        }));
    }

    let mut out_shape = a.shape_cloned();
    out_shape[axis] = 1;

    let mut out = Tensor::alloc_on(out_shape, backend);

    let (out_storage, out_layout) = out.storage_mut_and_layout();
    backend.reduce(
        ReductionOp::Prod,
        a.storage(),
        a.layout(),
        axis,
        out_storage,
        out_layout,
    )?;

    Ok(out)
}

/// Global product of all elements.
///
/// Equivalent to `torch.prod(input)`.
///
/// # Examples
///
/// ```
/// use coeus_core::SequentialBackend;
/// use coeus_ops::prod_tensor;
/// use coeus_tensor::Tensor;
///
/// let backend = SequentialBackend::new();
/// let input = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
/// let result = prod_tensor(&input, &backend).expect("valid product inputs");
/// assert_eq!(result.as_slice(), &[24.0]);
/// ```
pub fn prod_tensor<T: Scalar, B: BackendOps<T> + Default>(
    a: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let mut reduced = if a.ndim() == 0 {
        a.reshape([1])
    } else {
        a.clone()
    };

    for axis in 0..a.ndim() {
        reduced = prod_axis(&reduced, axis, backend)?;
    }

    Ok(reduced.reshape([1]))
}

/// Global product of all elements, returning its final scalar value.
///
/// The reduction stays on the selected backend; only the one-element result
/// crosses the backend boundary.
#[inline]
pub fn prod<T: Scalar, B: BackendOps<T> + Default>(a: &Tensor<T, B>, backend: &B) -> T {
    let reduced = prod_tensor(a, backend).expect("prod: provider reduction failed");
    let mut scalar = [T::zero()];
    backend.copy_to_host(reduced.storage(), &mut scalar);
    scalar[0]
}
