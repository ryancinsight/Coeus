//! Matrix multiplication sub-trait.
//!
//! [`MatmulOps`] is the interface-segregated sub-trait for all matmul kernel
//! dispatch.  The default `matmul_accumulate` and `batched_matmul_accumulate`
//! implementations call [`ElementwiseOps::elementwise_binary`], so those
/// defaults carry a `where Self: ElementwiseOps<T>` clause.
use coeus_core::{ComputeBackend, Layout, Scalar};

use super::super::defaults;
use super::elementwise::ElementwiseOps;

/// Matrix multiplication operations.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `MatmulOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// [`BackendOps`]: super::super::BackendOps
pub trait MatmulOps<T: Scalar>: ComputeBackend {
    /// Matrix multiplication.
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Matrix multiplication with accumulation: `c += a * b`.
    ///
    /// The default uses a temporary buffer and `elementwise_binary(Add)`,
    /// requiring [`ElementwiseOps`].  Backends with a fused accumulate
    /// override this method without the cross-trait bound.
    fn matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: ElementwiseOps<T>,
    {
        defaults::matmul::matmul_accumulate(self, a, a_layout, b, b_layout, c, c_layout)
    }

    /// Rank-3 batched matrix multiplication.
    fn batched_matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        defaults::matmul::batched_matmul(self, a, a_layout, b, b_layout, c, c_layout)
    }

    /// Rank-3 batched matrix multiplication with accumulation: `c += a * b`.
    ///
    /// The default uses a temporary buffer and `elementwise_binary(Add)`,
    /// requiring [`ElementwiseOps`].  Backends with a fused accumulate
    /// override this method without the cross-trait bound.
    fn batched_matmul_accumulate(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: ElementwiseOps<T>,
    {
        defaults::matmul::batched_matmul_accumulate(self, a, a_layout, b, b_layout, c, c_layout)
    }
}
