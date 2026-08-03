use crate::HephaestusProvider;
use coeus_core::Scalar;
use hephaestus_core::ElementwiseOps;

/// Provider-owned operations required by half-vector rotation.
pub trait RotateHalfProvider<T>: HephaestusProvider
where
    T: Scalar,
{
    /// Monomorphized elementwise operation bundle selected by this provider.
    type Operations: ElementwiseOps<Self::Device, T> + Default;
}
