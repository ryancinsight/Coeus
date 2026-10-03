//! The Coeus backend type generic over one Hephaestus provider, and its
//! `ComputeBackend` implementation; the per-concern operation traits are
//! implemented beside their kernels.

use crate::{error::HephaestusBackendError, storage::HephaestusStorage, HephaestusProvider};
use coeus_core::{ComputeBackend, Scalar, StorageMut};
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use std::future::Ready;

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

impl<P> ComputeBackend for HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    type Error = HephaestusBackendError;
    type DeviceBuffer<T: Scalar> = HephaestusStorage<P, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = Ready<T>;

    fn name(&self) -> &'static str {
        P::NAME
    }

    fn num_threads(&self) -> usize {
        1
    }

    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        HephaestusStorage::uninitialized(len)
    }

    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        HephaestusStorage::new(len)
    }

    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let values = vec![val; dst.buffer().len()];
        self.copy_to_device(&values, dst);
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        dst.make_unique();
        P::device()
            .write_buffer(dst.buffer(), src)
            .expect("Hephaestus host-to-device copy failed");
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        P::device()
            .download(src.buffer(), dst)
            .expect("Hephaestus device-to-host copy failed");
    }
}
