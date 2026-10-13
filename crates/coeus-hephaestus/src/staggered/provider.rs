use crate::{HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use hephaestus_core::Staggered3DOps;

/// Provider-owned staggered gradient/divergence marker.
///
/// A provider implements this when its device has the staggered kernels. There
/// is no scalar parameter: the provider states the pair in `f32`, because WGSL
/// does not guarantee `f64` storage and a generic scalar at this boundary would
/// be falsely generic.
pub trait StaggeredProvider: HephaestusProvider {
    /// Monomorphized Hephaestus staggered operations selected by this provider.
    type Operations: Staggered3DOps<Self::Device> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus staggered provider.
pub trait StaggeredBackend: HephaestusSeamBackend {
    /// Monomorphized Hephaestus operations for the selected device.
    type Operations: Staggered3DOps<Self::Device> + Default;
}

impl<P> StaggeredBackend for HephaestusBackend<P>
where
    P: StaggeredProvider,
{
    type Operations = P::Operations;
}
