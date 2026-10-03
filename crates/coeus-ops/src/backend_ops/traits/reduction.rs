//! Reduction sub-trait.
//!
//! [`ReductionOps`] is the interface-segregated sub-trait for all reduction
//! kernel dispatch (reduce, mean, argmax, argmin, topk, cumsum, suffix_sum,
//! cumprod, suffix_prod). The argmax/argmin/topk defaults are CPU-only and route to
//! Leto through [`super::super::CpuBackend`]. Cumulative scan methods are
//! required provider operations; no host-staging default is available.

use coeus_core::{ComputeBackend, FloatElement, Layout, Scalar};

use super::super::defaults;
use super::super::ops::ReductionOp;
use super::super::CpuBackend;

/// Reduction operations along an axis.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `ReductionOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// [`BackendOps`]: super::super::BackendOps
pub trait ReductionOps<T: Scalar>: ComputeBackend {
    /// Reduction operations along an axis.
    ///
    /// The arithmetic mean is not a [`ReductionOp`]; it is
    /// [`ReductionOps::mean`], bounded on `FloatElement`.
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout validation, provider
    /// execution, or output dispatch fails.
    fn reduce(
        &self,
        op: ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Arithmetic mean along an axis.
    ///
    /// The `FloatElement` bound makes integer mean unrepresentable: integer
    /// division would truncate the quotient. Every provider implements it
    /// against its native mean kernel, so no provider, CPU or accelerator,
    /// can be asked for an integer mean:
    ///
    /// ```compile_fail,E0277
    /// use coeus_core::Layout;
    /// use coeus_ops::ReductionOps;
    ///
    /// fn integer_mean<B: ReductionOps<i32>>(
    ///     backend: &B,
    ///     a: &B::DeviceBuffer<i32>,
    ///     layout: &Layout,
    ///     c: &mut B::DeviceBuffer<i32>,
    /// ) {
    ///     let _ = backend.mean(a, layout, 0, c, layout);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the backend-associated error when layout validation, provider
    /// execution, or output dispatch fails.
    fn mean(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: FloatElement;

    /// Compute the indices of the maximum values along `axis`.
    fn argmax(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<i64>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
        Self: CpuBackend,
    {
        defaults::reductions::argmax(self, a, a_layout, axis, c, c_layout)
    }

    /// Compute the indices of the minimum values along `axis`.
    fn argmin(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<i64>,
        c_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
        Self: CpuBackend,
    {
        defaults::reductions::argmin(self, a, a_layout, axis, c, c_layout)
    }

    /// Return the `k` largest (or smallest) values and their indices along an axis.
    #[expect(
        clippy::too_many_arguments,
        reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
    )]
    fn topk(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        k: usize,
        axis: usize,
        largest: bool,
        values: &mut Self::DeviceBuffer<T>,
        values_layout: &Layout,
        indices: &mut Self::DeviceBuffer<i64>,
        indices_layout: &Layout,
    ) where
        T: leto_ops::Scalar,
        Self: CpuBackend,
    {
        defaults::reductions::topk(
            self,
            a,
            a_layout,
            k,
            axis,
            largest,
            values,
            values_layout,
            indices,
            indices_layout,
        )
    }

    /// Inclusive cumulative sum along an axis through the selected provider.
    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar;

    /// Inclusive cumulative suffix sum through the selected provider.
    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar;

    /// Inclusive cumulative product through the selected provider.
    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar;

    /// Inclusive cumulative suffix product through the selected provider.
    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar;
}
