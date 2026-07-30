use crate::HephaestusProvider;
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
