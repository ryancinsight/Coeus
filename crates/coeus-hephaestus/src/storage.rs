use crate::reduction::HephaestusProvider;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_core::{ComputeDevice, DeviceBuffer};
use std::{marker::PhantomData, sync::Arc};
use themis::{MemoryTier, PlacementHint};

/// Reference-counted Coeus storage backed by one Hephaestus device buffer.
pub struct HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod,
{
    pub(crate) buffer: Arc<<P::Device as ComputeDevice>::Buffer<T>>,
    marker: PhantomData<P>,
}

impl<P, T> Clone for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod,
{
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            marker: PhantomData,
        }
    }
}

impl<P, T> HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar,
{
    /// Adopt an initialized provider buffer without copying its contents.
    #[must_use]
    pub fn from_buffer(buffer: <P::Device as ComputeDevice>::Buffer<T>) -> Self {
        Self {
            buffer: Arc::new(buffer),
            marker: PhantomData,
        }
    }

    /// Adopt a shared provider buffer handle without copying its contents.
    ///
    /// Vendor crates that keep their own storage wrapper (for example Coeus
    /// CUDA's `CudaStorage`) can share the same `Arc` handle with the bridge,
    /// so the generic backend implementations dispatch over the identical
    /// device buffer with zero copies and no re-allocation.
    #[must_use]
    pub fn from_arc(buffer: Arc<<P::Device as ComputeDevice>::Buffer<T>>) -> Self {
        Self {
            buffer,
            marker: PhantomData,
        }
    }

    /// Allocate zeroed storage in the provider's device tier.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let buffer = P::device()
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Hephaestus provider allocation failed");
        Self::from_buffer(buffer)
    }

    pub(crate) fn uninitialized(len: usize) -> Self {
        let buffer = P::device()
            .alloc_uninitialized_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Hephaestus provider allocation failed");
        Self::from_buffer(buffer)
    }

    /// Borrow the typed Hephaestus buffer for provider dispatch.
    #[must_use]
    pub fn buffer(&self) -> &<P::Device as ComputeDevice>::Buffer<T> {
        self.buffer.as_ref()
    }
}

impl<P, T> coeus_core::storage::private::Sealed for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod,
{
}

// SAFETY: `HephaestusProvider` requires its device buffers to be safe to move
// between threads while the provider owns the device synchronization contract.
unsafe impl<P, T> Send for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod + Send,
{
}

// SAFETY: `HephaestusProvider` requires shared buffer handles to be safe to
// retain behind an Arc; mutable access remains mediated by `StorageMut`.
unsafe impl<P, T> Sync for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: eunomia::Pod + Sync,
{
}

impl<P, T> Storage<T> for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar,
{
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<P, T> StorageMut<T> for HephaestusStorage<P, T>
where
    P: HephaestusProvider,
    T: Scalar,
{
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) <= 1 {
            return;
        }
        // COW detachment is a storage operation, so preserve the provider's
        // allocation tier and keep the full payload on-device. The device
        // copy overwrites every element before the detached buffer is exposed,
        // so the replacement does not require a redundant initialization pass.
        // The `StorageMut` contract is infallible; provider failures therefore
        // remain explicit invariant failures until that upstream contract is
        // made fallible.
        let device = P::device();
        let replacement = device
            .alloc_uninitialized_with_hint(
                self.buffer.len(),
                PlacementHint::Tier(self.buffer.tier()),
            )
            .expect("Hephaestus storage uniqueness allocation failed");
        device
            .copy_buffer(self.buffer.as_ref(), &replacement)
            .expect("Hephaestus storage uniqueness device copy failed");
        self.buffer = Arc::new(replacement);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduction::HephaestusProvider;
    use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
    use std::{
        marker::PhantomData,
        sync::atomic::{AtomicUsize, Ordering},
        sync::{Arc, Mutex},
    };

    static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
    static DEVICE_COPIES: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestInitialization {
        Zeroed,
        Uninitialized,
    }

    #[derive(Debug, Clone)]
    struct TestBuffer<T: eunomia::Pod> {
        bytes: Arc<Mutex<Vec<u8>>>,
        len: usize,
        tier: MemoryTier,
        initialization: TestInitialization,
        marker: PhantomData<T>,
    }

    impl<T: eunomia::Pod> DeviceBuffer<T> for TestBuffer<T> {
        fn len(&self) -> usize {
            self.len
        }

        fn tier(&self) -> MemoryTier {
            self.tier
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct TestProvider;

    #[derive(Debug, Clone, Copy, Default)]
    struct TestDevice;

    fn byte_len<T: eunomia::Pod>(len: usize) -> hephaestus_core::Result<usize> {
        len.checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: "test buffer size overflow".to_owned(),
            })
    }

    fn empty_buffer<T: eunomia::Pod>(
        len: usize,
        tier: MemoryTier,
        initialization: TestInitialization,
    ) -> hephaestus_core::Result<TestBuffer<T>> {
        let initial_byte = match initialization {
            TestInitialization::Zeroed => 0,
            TestInitialization::Uninitialized => 0xa5,
        };
        Ok(TestBuffer {
            bytes: Arc::new(Mutex::new(vec![initial_byte; byte_len::<T>(len)?])),
            len,
            tier,
            initialization,
            marker: PhantomData,
        })
    }

    fn require_len<T: eunomia::Pod>(
        buffer: &TestBuffer<T>,
        len: usize,
    ) -> hephaestus_core::Result<()> {
        if buffer.len == len {
            Ok(())
        } else {
            Err(HephaestusError::LengthMismatch {
                host_len: len,
                device_len: buffer.len,
            })
        }
    }

    impl ComputeDevice for TestDevice {
        type Buffer<T: eunomia::Pod> = TestBuffer<T>;

        fn backend_name(&self) -> &'static str {
            "test"
        }

        fn topology(&self) -> Option<&themis::GpuTopology> {
            None
        }

        fn alloc_zeroed_with_hint<T: eunomia::Pod>(
            &self,
            len: usize,
            hint: PlacementHint,
        ) -> hephaestus_core::Result<Self::Buffer<T>> {
            empty_buffer(
                len,
                match hint {
                    PlacementHint::Tier(tier) => tier,
                    _ => MemoryTier::Device,
                },
                TestInitialization::Zeroed,
            )
        }

        fn alloc_uninitialized_with_hint<T: eunomia::Pod>(
            &self,
            len: usize,
            hint: PlacementHint,
        ) -> hephaestus_core::Result<Self::Buffer<T>> {
            empty_buffer(
                len,
                match hint {
                    PlacementHint::Tier(tier) => tier,
                    _ => MemoryTier::Device,
                },
                TestInitialization::Uninitialized,
            )
        }

        fn upload_with_hint<T: eunomia::Pod>(
            &self,
            host: &[T],
            hint: PlacementHint,
        ) -> hephaestus_core::Result<Self::Buffer<T>> {
            let buffer = self.alloc_zeroed_with_hint(host.len(), hint)?;
            self.write_buffer(&buffer, host)?;
            Ok(buffer)
        }

        fn download<T: eunomia::Pod>(
            &self,
            buffer: &Self::Buffer<T>,
            out: &mut [T],
        ) -> hephaestus_core::Result<()> {
            DOWNLOADS.fetch_add(1, Ordering::Relaxed);
            require_len(buffer, out.len())?;
            let bytes = buffer
                .bytes
                .lock()
                .map_err(|_| HephaestusError::TransferFailed {
                    message: "test buffer lock poisoned".to_owned(),
                })?;
            eunomia::layout::cast_slice_mut(out).copy_from_slice(&bytes);
            Ok(())
        }

        fn write_buffer<T: eunomia::Pod>(
            &self,
            buffer: &Self::Buffer<T>,
            host: &[T],
        ) -> hephaestus_core::Result<()> {
            require_len(buffer, host.len())?;
            let mut bytes = buffer
                .bytes
                .lock()
                .map_err(|_| HephaestusError::TransferFailed {
                    message: "test buffer lock poisoned".to_owned(),
                })?;
            bytes.copy_from_slice(eunomia::layout::cast_slice(host));
            Ok(())
        }

        fn write_sub_buffer<T: eunomia::Pod>(
            &self,
            buffer: &Self::Buffer<T>,
            offset: usize,
            host: &[T],
        ) -> hephaestus_core::Result<()> {
            let end = offset
                .checked_add(host.len())
                .ok_or(HephaestusError::LengthMismatch {
                    host_len: host.len(),
                    device_len: buffer.len,
                })?;
            if end > buffer.len {
                return Err(HephaestusError::LengthMismatch {
                    host_len: end,
                    device_len: buffer.len,
                });
            }
            let mut bytes = buffer
                .bytes
                .lock()
                .map_err(|_| HephaestusError::TransferFailed {
                    message: "test buffer lock poisoned".to_owned(),
                })?;
            let start_bytes = offset * std::mem::size_of::<T>();
            let host_bytes = eunomia::layout::cast_slice(host);
            bytes[start_bytes..start_bytes + host_bytes.len()].copy_from_slice(host_bytes);
            Ok(())
        }

        fn copy_buffer<T: eunomia::Pod>(
            &self,
            src: &Self::Buffer<T>,
            dst: &Self::Buffer<T>,
        ) -> hephaestus_core::Result<()> {
            require_len(dst, src.len)?;
            let src_bytes = src
                .bytes
                .lock()
                .map_err(|_| HephaestusError::TransferFailed {
                    message: "test source lock poisoned".to_owned(),
                })?;
            let mut dst_bytes = dst
                .bytes
                .lock()
                .map_err(|_| HephaestusError::TransferFailed {
                    message: "test destination lock poisoned".to_owned(),
                })?;
            dst_bytes.copy_from_slice(&src_bytes);
            DEVICE_COPIES.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn synchronize(&self) -> hephaestus_core::Result<()> {
            Ok(())
        }
    }

    // SAFETY: The test device uses Arc<Mutex<_>> for buffer storage and has no
    // thread-affine state, satisfying the provider buffer ownership contract.
    unsafe impl HephaestusProvider for TestProvider {
        type Device = TestDevice;

        const NAME: &'static str = "test";

        fn device() -> &'static Self::Device {
            static DEVICE: TestDevice = TestDevice;
            &DEVICE
        }

        fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
            Ok(Self::device())
        }
    }

    #[test]
    fn backend_routes_allocation_by_initialization_contract() {
        use coeus_core::ComputeBackend;

        let backend = crate::reduction::HephaestusBackend::<TestProvider>::new();
        let scratch = backend.allocate::<u32>(4);
        assert_eq!(
            scratch.buffer.initialization,
            TestInitialization::Uninitialized
        );
        let mut poison = [0; 4];
        backend.copy_to_host(&scratch, &mut poison);
        assert_eq!(poison, [0xa5a5_a5a5; 4]);

        let zeroed = backend.allocate_zeroed::<u32>(4);
        assert_eq!(zeroed.buffer.initialization, TestInitialization::Zeroed);
        let mut values = [u32::MAX; 4];
        backend.copy_to_host(&zeroed, &mut values);
        assert_eq!(values, [0; 4]);
    }

    #[test]
    fn make_unique_copies_device_data_without_host_download() {
        DOWNLOADS.store(0, Ordering::Relaxed);
        DEVICE_COPIES.store(0, Ordering::Relaxed);

        let device = TestProvider::device();
        let mut storage = HephaestusStorage::<TestProvider, u32>::new(4);
        device
            .write_buffer(storage.buffer.as_ref(), &[1, 2, 3, 4])
            .expect("write test storage");
        let shared = storage.clone();
        let downloads_before = DOWNLOADS.load(Ordering::Relaxed);

        StorageMut::make_unique(&mut storage);

        assert_eq!(DOWNLOADS.load(Ordering::Relaxed), downloads_before);
        assert_eq!(DEVICE_COPIES.load(Ordering::Relaxed), 1);

        let mut detached = [0; 4];
        let mut retained = [0; 4];
        device
            .download(storage.buffer.as_ref(), &mut detached)
            .expect("read detached storage");
        device
            .download(shared.buffer.as_ref(), &mut retained)
            .expect("read retained storage");
        assert_eq!(detached, [1, 2, 3, 4]);
        assert_eq!(retained, [1, 2, 3, 4]);
        assert_eq!(storage.buffer.tier(), MemoryTier::Device);
        assert_eq!(shared.buffer.tier(), MemoryTier::Device);
    }
}
