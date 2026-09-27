use crate::HephaestusProvider;
use coeus_core::Scalar;
use hephaestus_core::CrossProductOps;

/// Provider-owned operations required by the on-device cross-product bridge.
///
/// Implemented only where hephaestus ships `CrossProductOps` (host/CPU and
/// WGPU as of ADR 0077 — CUDA, ROCm, and Metal do not yet). A provider
/// without this impl simply has no `CrossProductProvider<T>` bound to
/// satisfy; its backend's `coeus_ops::CrossOps` implementation calls the
/// shared host-fold path directly instead (`coeus_ops::cross_fold`).
pub trait CrossProductProvider<T>: HephaestusProvider
where
    T: Scalar,
{
    /// Monomorphized cross-product operation bundle selected by this
    /// provider.
    type Operations: CrossProductOps<Self::Device, T> + Default;
}
