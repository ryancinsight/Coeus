use crate::{HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use coeus_core::Scalar;
use hephaestus_core::PoolingOps;

/// Provider-owned pooling operation marker.
///
/// # Safety
///
/// Successful forward operations selected by `Operations` must initialize
/// every logical output element without reading prior contents. Backward
/// operations accumulate and may read their gradient destinations, which
/// must be initialized before dispatch. A dispatch error may leave an output
/// partially written.
pub unsafe trait PoolingProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus pooling operations selected by this provider.
    type Operations: PoolingOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus pooling provider.
pub trait PoolingBackend<T>: HephaestusSeamBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operations for the selected device.
    type Operations: PoolingOps<Self::Device, T> + Default;
}

impl<P, T> PoolingBackend<T> for HephaestusBackend<P>
where
    P: PoolingProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Operations = P::Operations;
}
