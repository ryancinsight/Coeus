use crate::HephaestusProvider;
use coeus_core::Scalar;
use hephaestus_core::RandomInitOps;

/// Provider-owned random-initialization operation marker.
pub trait RandomInitProvider<T>: HephaestusProvider
where
    T: Scalar,
{
    /// Monomorphized Hephaestus operation selected by this provider.
    type Operations: RandomInitOps<Self::Device, T> + Default;
}
