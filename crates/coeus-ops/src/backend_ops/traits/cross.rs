use coeus_core::{ComputeBackend, Layout, Scalar};

/// Backend-selected per-channel 3-vector cross product.
///
/// Standalone capability, not a [`BackendOps`](super::super::BackendOps)
/// bundle member: every backend that exists today already computes
/// `coeus_ops::cross` through the shared host-fold path
/// (`crate::backend_ops::defaults::cross::cross_fold`), so folding this
/// into the required bundle would force an identical no-op impl onto every
/// backend for a capability `dot`/`cross` never required before. A backend
/// with an on-device seam (currently `WgpuBackend`, for the contiguous
/// last-axis layout) overrides [`CrossOps::cross_storage`] instead of
/// relying on the shared default.
pub trait CrossOps<T: Scalar>: ComputeBackend {
    /// Allocate and initialize storage for the cross product of `a` and `b`
    /// along `dim`.
    ///
    /// `a` and `b` share `a_layout`'s shape; the caller (`coeus_ops::cross`)
    /// has already asserted `a_layout.shape()[dim] == 3` and that `a`/`b`
    /// agree in shape — this method assumes both hold.
    ///
    /// # Errors
    ///
    /// Returns a typed backend failure for allocation or provider dispatch
    /// failure.
    fn cross_storage(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        dim: usize,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error>;
}
