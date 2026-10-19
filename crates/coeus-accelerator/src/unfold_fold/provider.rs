use crate::{HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use coeus_core::Scalar;
use hephaestus_core::SlidingWindowOps;

/// Provider-owned sliding-window operation marker.
///
/// # Safety
///
/// Successful unfold operations selected by `Operations` must initialize
/// every logical column output without reading prior contents. Fold operations
/// must zero their spatial output before accumulating. A dispatch error may
/// leave an output partially written.
pub unsafe trait UnfoldFoldProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus sliding-window operations selected by this provider.
    type Operations: SlidingWindowOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus sliding-window provider.
pub trait UnfoldFoldBackend<T>: HephaestusSeamBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operations for the selected device.
    type Operations: SlidingWindowOps<Self::Device, T> + Default;
}

impl<P, T> UnfoldFoldBackend<T> for HephaestusBackend<P>
where
    P: UnfoldFoldProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Operations = P::Operations;
}
