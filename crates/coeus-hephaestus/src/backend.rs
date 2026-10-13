//! Generic backend selector and provider seam over one Hephaestus device.
//!
//! This module owns the [`HephaestusProvider`] device-acquisition trait and the
//! [`HephaestusBackend`] type with its [`ComputeBackend`] implementation.
//! Operation dispatch lives in the per-operation modules; they are generic over
//! the provider selected here.

use crate::{error::HephaestusBackendError, storage::HephaestusStorage};
use coeus_core::{BackendError, ComputeBackend, Scalar, StorageMut};
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use std::future::Ready;

/// Common provider identity and device acquisition seam.
///
/// # Safety
///
/// An implementation must ensure that every typed buffer exposed by its
/// [`hephaestus_core::ComputeDevice`] remains safe to retain behind an
/// `Arc` and to move between Coeus worker threads. Provider kernels must also
/// synchronize access according to their device API's contract.
pub unsafe trait HephaestusProvider: Send + Sync + Clone + Copy + Default + 'static {
    /// Concrete Hephaestus device type selected by this provider.
    type Device: ComputeDevice + Send + Sync + 'static;

    /// Coeus error type used by this provider's storage and dispatch boundary.
    type Error: std::error::Error
        + From<coeus_core::BackendError>
        + From<HephaestusBackendError>
        + Send
        + Sync
        + 'static;

    /// Stable backend name used by Coeus diagnostics.
    const NAME: &'static str;

    /// Return the lazily acquired device owned by this provider.
    fn device() -> &'static Self::Device;

    /// Try to acquire the provider device without panicking.
    ///
    /// # Errors
    ///
    /// Returns the provider's typed acquisition failure.
    fn try_device() -> hephaestus_core::Result<&'static Self::Device>;
}

/// Generic Coeus backend implementation over one Hephaestus provider.
#[derive(Debug)]
pub struct HephaestusBackend<P>(std::marker::PhantomData<P>);

impl<P> Copy for HephaestusBackend<P> {}

impl<P> Clone for HephaestusBackend<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Default for HephaestusBackend<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> HephaestusBackend<P> {
    /// Construct the zero-sized generic backend selector.
    #[must_use]
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

// SAFETY: provider storage never exposes host slices, and successful fill and
// transfer operations initialize their complete destination buffers.
unsafe impl<P> ComputeBackend for HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    type Error = P::Error;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<P, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = Ready<T>;

    fn name(&self) -> &'static str {
        P::NAME
    }

    fn num_threads(&self) -> usize {
        1
    }

    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // SAFETY: the caller of this unsafe backend method must initialize the
        // returned buffer before any read.
        unsafe { HephaestusStorage::uninitialized(len) }
            .map_err(|source| P::Error::from(HephaestusBackendError::device("allocate", source)))
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        HephaestusStorage::new(len).map_err(|source| {
            P::Error::from(HephaestusBackendError::device("allocate_zeroed", source))
        })
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error> {
        let values = vec![val; dst.buffer().len()];
        self.copy_to_device(&values, dst)
    }

    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.buffer().len() {
            return Err(BackendError::BufferLengthMismatch {
                operation: "copy_to_device",
                source_len: src.len(),
                destination_len: dst.buffer().len(),
            }
            .into());
        }
        let device = P::try_device().map_err(|source| {
            P::Error::from(HephaestusBackendError::device("copy_to_device", source))
        })?;
        dst.make_unique()?;
        device.write_buffer(dst.buffer(), src).map_err(|source| {
            P::Error::from(HephaestusBackendError::device("copy_to_device", source))
        })
    }

    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        if src.buffer().len() != dst.len() {
            return Err(BackendError::BufferLengthMismatch {
                operation: "copy_to_host",
                source_len: src.buffer().len(),
                destination_len: dst.len(),
            }
            .into());
        }
        P::try_device()
            .map_err(|source| {
                P::Error::from(HephaestusBackendError::device("copy_to_host", source))
            })?
            .download(src.buffer(), dst)
            .map_err(|source| {
                P::Error::from(HephaestusBackendError::device("copy_to_host", source))
            })
    }
}
