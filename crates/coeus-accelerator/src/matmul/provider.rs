use crate::{HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use coeus_core::Scalar;
use hephaestus_core::DenseProductOps;

/// Provider-owned scalar dense-product operation marker.
///
/// # Safety
///
/// Every successful overwrite operation selected by `Operations` must
/// initialize every logical output element without reading the output's prior
/// contents. A dispatch error may leave an output partially written.
pub unsafe trait MatmulProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operation marker selected by this provider.
    type Operations: DenseProductOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus dense-product
/// provider.
///
/// Implementations expose the selected device buffer without host transfer and
/// preserve backend-specific error types at the consumer boundary. A backend
/// that is not [`HephaestusBackend`] implements this trait directly and reuses
/// [`crate::matmul`] rather than carrying its own kernel.
pub trait MatmulBackend<T>: HephaestusSeamBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operations for the selected device.
    type Operations: DenseProductOps<Self::Device, T> + Default;
}

impl<P, T> MatmulBackend<T> for HephaestusBackend<P>
where
    P: MatmulProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Operations = P::Operations;
}
