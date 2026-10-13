//! Shared device-access seam for the per-operation backend traits.
//!
//! The seam-family operation traits ([`ConvolutionBackend`](crate::ConvolutionBackend),
//! [`MatmulBackend`](crate::MatmulBackend), [`PoolingBackend`](crate::PoolingBackend),
//! [`StaggeredBackend`](crate::StaggeredBackend), and
//! [`UnfoldFoldBackend`](crate::UnfoldFoldBackend)) each bound a backend to one
//! Hephaestus operation set. They all need the same four capabilities — the
//! selected device, borrowing the provider buffer behind Coeus storage, and
//! mapping configuration and dispatch failures into the backend's typed error —
//! so those live once here as a supertrait. Each operation trait keeps only its
//! monomorphized `Operations` marker, and every backend implements this seam
//! once no matter how many operations it supports.

use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar};
use hephaestus_core::{ComputeDevice, HephaestusError};

/// Zero-cost device access shared by every seam-family operation backend.
///
/// A backend that is not [`HephaestusBackend`] implements this trait once,
/// directly, alongside whichever per-operation traits it supports.
pub trait HephaestusSeamBackend: ComputeBackend {
    /// Concrete Hephaestus device selected by this backend.
    type Device: ComputeDevice + Send + Sync + 'static;

    /// Return the lazily acquired provider device.
    fn device() -> &'static Self::Device;

    /// Borrow the provider buffer contained by Coeus storage.
    fn buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T>;

    /// Map parameter or layout rejection into the backend's typed error.
    fn configuration_error(operation: &'static str, reason: String) -> Self::Error;

    /// Map a Hephaestus provider failure into the backend's typed error.
    fn dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error;
}

impl<P> HephaestusSeamBackend for HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    type Device = P::Device;

    fn device() -> &'static Self::Device {
        P::device()
    }

    fn buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn configuration_error(operation: &'static str, reason: String) -> Self::Error {
        P::Error::from(crate::error::invalid_configuration_error(operation, reason))
    }

    fn dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        P::Error::from(HephaestusBackendError::device(operation, source))
    }
}
