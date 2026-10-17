use crate::{HephaestusBackend, HephaestusProvider, HephaestusSeamBackend};
use coeus_core::Scalar;
use hephaestus_core::ConvolutionOps;

/// Provider-owned scalar convolution operation marker.
pub trait ConvolutionProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operation marker selected by this provider.
    type Operations: ConvolutionOps<Self::Device, T> + Default;
}

/// Zero-cost binding from a Coeus backend to one Hephaestus convolution
/// provider.
///
/// Implementations expose the selected device buffer without host transfer and
/// preserve backend-specific error types at the consumer boundary.
pub trait ConvolutionBackend<T>: HephaestusSeamBackend
where
    T: Scalar + leto_ops::Scalar,
{
    /// Monomorphized Hephaestus operations for the selected device.
    type Operations: ConvolutionOps<Self::Device, T> + Default;
}

impl<P, T> ConvolutionBackend<T> for HephaestusBackend<P>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    type Operations = P::Operations;
}
