use coeus_core::{ComputeBackend, Scalar};

use super::traits::{
    AttentionOps, ConvOps, ElementwiseOps, MatmulOps, OptimizerOps, PoolOps, ReductionOps,
    UnfoldFoldOps,
};

/// Dynamic operations supported by execution hardware backends.
///
/// `BackendOps<T>` is the single dispatch surface that routes all tensor kernels
/// (elementwise, matmul, conv, pooling, attention, optimizer steps, unfold/fold)
/// to the underlying device.  It is a **super-trait** composed of eight
/// interface-segregated sub-traits (see the [`super::traits`] module).  Backends
/// implement each sub-trait independently; the blanket impl below provides
/// `BackendOps` automatically.
///
/// The CPU path is provided by sub-trait impls in `backend_ops::cpu_impl`;
/// other devices add new sub-trait impls without touching the algorithm
/// bodies.
///
/// # Examples
///
/// `SequentialBackend` implements `BackendOps` and drives all kernel dispatch:
///
/// ```
/// use coeus_ops::{BackendOps, add};
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// fn compute<B: BackendOps<f32>>(
///     a: &Tensor<f32, B>,
///     b: &Tensor<f32, B>,
///     backend: &B,
/// ) -> Result<Tensor<f32, B>, B::Error> {
///     add(a, b, backend)
/// }
///
/// let backend = SequentialBackend::new();
/// let a = Tensor::<f32, SequentialBackend>::from_slice([2], &[1.0, 2.0]).expect("construct tensor");
/// let b = Tensor::<f32, SequentialBackend>::from_slice([2], &[3.0, 4.0]).expect("construct tensor");
/// let c = compute(&a, &b, &backend).expect("dispatch operation");
/// assert_eq!(c.as_slice(), &[4.0, 6.0]);
/// ```
pub trait BackendOps<T: Scalar>:
    ComputeBackend
    + ElementwiseOps<T>
    + MatmulOps<T>
    + ReductionOps<T>
    + ConvOps<T>
    + PoolOps<T>
    + AttentionOps<T>
    + OptimizerOps<T>
    + UnfoldFoldOps<T>
{
}

/// Blanket impl: any backend that implements all eight sub-traits automatically
/// satisfies `BackendOps`.  No additional methods are required — `BackendOps`
/// is a marker super-trait.
impl<T: Scalar, B> BackendOps<T> for B where
    B: ComputeBackend
        + ElementwiseOps<T>
        + MatmulOps<T>
        + ReductionOps<T>
        + ConvOps<T>
        + PoolOps<T>
        + AttentionOps<T>
        + OptimizerOps<T>
        + UnfoldFoldOps<T>
{
}
