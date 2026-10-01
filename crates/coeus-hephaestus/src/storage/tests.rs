use super::*;
use crate::reduction::HephaestusProvider;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use std::{
    cell::Cell,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

std::thread_local! {
    static DOWNLOADS: Cell<usize> = const { Cell::new(0) };
    static DEVICE_COPIES: Cell<usize> = const { Cell::new(0) };
    static FAIL_UNINITIALIZED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static FAIL_DEVICE_COPIES: Cell<usize> = const { Cell::new(0) };
}

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
struct UnavailableProvider;

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

fn require_len<T: eunomia::Pod>(buffer: &TestBuffer<T>, len: usize) -> hephaestus_core::Result<()> {
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
        if FAIL_UNINITIALIZED_ALLOCATIONS.with(|failure| failure.replace(0) != 0) {
            return Err(HephaestusError::AllocationFailed {
                message: "injected COW allocation failure".to_owned(),
            });
        }
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
        DOWNLOADS.with(|count| count.set(count.get() + 1));
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
        if FAIL_DEVICE_COPIES.with(|failure| failure.replace(0) != 0) {
            return Err(HephaestusError::DispatchFailed {
                message: "injected device copy failure".to_owned(),
            });
        }
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
        DEVICE_COPIES.with(|count| count.set(count.get() + 1));
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
    type Error = crate::error::HephaestusBackendError;

    const NAME: &'static str = "test";

    fn device() -> &'static Self::Device {
        static DEVICE: TestDevice = TestDevice;
        &DEVICE
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        Ok(Self::device())
    }
}

// SAFETY: This provider exposes the same thread-safe test buffer type as
// `TestProvider`; its fallible acquisition path rejects every request before
// returning a device.
unsafe impl HephaestusProvider for UnavailableProvider {
    type Device = TestDevice;
    type Error = crate::error::HephaestusBackendError;

    const NAME: &'static str = "unavailable-test";

    fn device() -> &'static Self::Device {
        static DEVICE: TestDevice = TestDevice;
        &DEVICE
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        Err(HephaestusError::DeviceUnavailable {
            message: "injected unavailable test device".to_owned(),
        })
    }
}

fn assert_device_unavailable(error: crate::HephaestusBackendError, expected_operation: &str) {
    match error {
        crate::HephaestusBackendError::Device { operation, source } => {
            assert_eq!(operation, expected_operation);
            assert!(matches!(
                source,
                HephaestusError::DeviceUnavailable { message }
                    if message == "injected unavailable test device"
            ));
        }
        other => panic!("expected typed device acquisition failure, got {other}"),
    }
}

fn expect_backend_failure<T>(
    result: Result<T, crate::HephaestusBackendError>,
    expectation: &str,
) -> crate::HephaestusBackendError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("{expectation}"),
    }
}

#[test]
fn backend_routes_allocation_by_initialization_contract() {
    use coeus_core::ComputeBackend;

    let backend = crate::reduction::HephaestusBackend::<TestProvider>::new();
    // SAFETY: the test fills every element before any safe read.
    let mut scratch =
        unsafe { backend.allocate::<u32>(4) }.expect("invariant: test backend allocation succeeds");
    assert_eq!(
        scratch.buffer.initialization,
        TestInitialization::Uninitialized
    );
    assert!(coeus_core::Storage::try_as_slice(&scratch).is_none());
    backend
        .fill(&mut scratch, 17)
        .expect("invariant: test backend fill succeeds");
    let mut values = [0; 4];
    backend
        .copy_to_host(&scratch, &mut values)
        .expect("invariant: test backend transfer succeeds");
    assert_eq!(values, [17; 4]);

    let zeroed = backend
        .allocate_zeroed::<u32>(4)
        .expect("invariant: test backend allocation succeeds");
    assert_eq!(zeroed.buffer.initialization, TestInitialization::Zeroed);
    let mut values = [u32::MAX; 4];
    backend
        .copy_to_host(&zeroed, &mut values)
        .expect("invariant: test backend transfer succeeds");
    assert_eq!(values, [0; 4]);
}

#[test]
fn backend_storage_operations_report_device_acquisition_failure() {
    use coeus_core::ComputeBackend;

    let Err(error) = HephaestusStorage::<UnavailableProvider, u32>::new(4) else {
        panic!("fallible storage construction must preserve provider acquisition failure");
    };
    assert!(matches!(
        error,
        HephaestusError::DeviceUnavailable { message }
            if message == "injected unavailable test device"
    ));

    let backend = crate::reduction::HephaestusBackend::<UnavailableProvider>::new();
    assert_device_unavailable(
        expect_backend_failure(
            // SAFETY: only the returned error is inspected; an unexpected
            // uninitialized buffer is dropped without being read.
            unsafe { backend.allocate::<u32>(4) },
            "unavailable provider must reject allocation",
        ),
        "allocate",
    );
    assert_device_unavailable(
        expect_backend_failure(
            backend.allocate_zeroed::<u32>(4),
            "unavailable provider must reject zeroed allocation",
        ),
        "allocate_zeroed",
    );

    let buffer = TestProvider::device()
        .alloc_zeroed_with_hint(4, PlacementHint::Tier(MemoryTier::Device))
        .expect("invariant: test buffer allocation succeeds");
    // SAFETY: the provider allocated the buffer with zeroed initialization.
    let mut storage = unsafe { HephaestusStorage::<UnavailableProvider, u32>::from_buffer(buffer) };

    assert_device_unavailable(
        expect_backend_failure(
            backend.fill_zero(&mut storage),
            "unavailable provider must reject zero fill",
        ),
        "copy_to_device",
    );
    assert_device_unavailable(
        expect_backend_failure(
            backend.fill(&mut storage, 7),
            "unavailable provider must reject fill",
        ),
        "copy_to_device",
    );
    assert_device_unavailable(
        expect_backend_failure(
            backend.copy_to_device(&[1, 2, 3, 4], &mut storage),
            "unavailable provider must reject upload",
        ),
        "copy_to_device",
    );
    let mut retained = [u32::MAX; 4];
    TestProvider::device()
        .download(storage.buffer(), &mut retained)
        .expect("invariant: test device can inspect rejected writes");
    assert_eq!(retained, [0; 4]);

    let mut host = [u32::MAX; 4];
    assert_device_unavailable(
        expect_backend_failure(
            backend.copy_to_host(&storage, &mut host),
            "unavailable provider must reject download",
        ),
        "copy_to_host",
    );
    assert_eq!(host, [u32::MAX; 4]);
}

#[test]
fn make_unique_copies_device_data_without_host_download() {
    DOWNLOADS.with(|count| count.set(0));
    DEVICE_COPIES.with(|count| count.set(0));

    let device = TestProvider::device();
    let mut storage = HephaestusStorage::<TestProvider, u32>::new(4)
        .expect("invariant: test provider allocation succeeds");
    device
        .write_buffer(storage.buffer.as_ref(), &[1, 2, 3, 4])
        .expect("write test storage");
    let shared = storage.clone();
    let downloads_before = DOWNLOADS.with(Cell::get);

    StorageMut::make_unique(&mut storage).expect("COW detachment succeeds");

    assert_eq!(DOWNLOADS.with(Cell::get), downloads_before);
    assert_eq!(DEVICE_COPIES.with(Cell::get), 1);

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

#[test]
fn make_unique_reports_provider_allocation_failure() {
    let device = TestProvider::device();
    let mut storage = HephaestusStorage::<TestProvider, u32>::new(4)
        .expect("invariant: test provider allocation succeeds");
    device
        .write_buffer(storage.buffer(), &[2, 3, 5, 7])
        .expect("invariant: initial test data write succeeds");
    let shared = storage.clone();
    let allocation_before = storage.allocation_id();
    FAIL_UNINITIALIZED_ALLOCATIONS.with(|failure| failure.set(1));

    let error = StorageMut::make_unique(&mut storage)
        .expect_err("injected COW allocation failure must remain observable");
    assert!(matches!(
        error,
        HephaestusBackendError::Device {
            operation: "storage uniqueness allocation",
            source: HephaestusError::AllocationFailed { message }
        } if message == "injected COW allocation failure"
    ));
    assert_eq!(storage.allocation_id(), allocation_before);
    assert_eq!(shared.allocation_id(), allocation_before);

    let mut retained = [0; 4];
    device
        .download(storage.buffer(), &mut retained)
        .expect("invariant: original allocation remains readable");
    assert_eq!(retained, [2, 3, 5, 7]);
}

#[test]
fn make_unique_reports_device_acquisition_failure_without_replacing_storage() {
    let buffer = TestProvider::device()
        .alloc_zeroed_with_hint(4, PlacementHint::Tier(MemoryTier::Device))
        .expect("invariant: test buffer allocation succeeds");
    TestProvider::device()
        .write_buffer(&buffer, &[13, 17, 19, 23])
        .expect("invariant: initial test data write succeeds");
    // SAFETY: the provider initialized all four elements before adoption.
    let mut storage = unsafe { HephaestusStorage::<UnavailableProvider, u32>::from_buffer(buffer) };
    let shared = storage.clone();
    let allocation_before = storage.allocation_id();

    let error = StorageMut::make_unique(&mut storage)
        .expect_err("unavailable provider must reject COW detachment");
    assert_device_unavailable(error, "storage uniqueness device acquisition");
    assert_eq!(storage.allocation_id(), allocation_before);
    assert_eq!(shared.allocation_id(), allocation_before);

    let mut retained = [0; 4];
    TestProvider::device()
        .download(storage.buffer(), &mut retained)
        .expect("invariant: source allocation remains readable");
    assert_eq!(retained, [13, 17, 19, 23]);
}

#[test]
fn make_unique_reports_provider_copy_failure_without_replacing_storage() {
    let device = TestProvider::device();
    let mut storage = HephaestusStorage::<TestProvider, u32>::new(4)
        .expect("invariant: test provider allocation succeeds");
    device
        .write_buffer(storage.buffer.as_ref(), &[3, 5, 7, 11])
        .expect("invariant: test buffer write succeeds");
    let retained = storage.clone();
    let allocation_before = storage.allocation_id();
    FAIL_DEVICE_COPIES.with(|failure| failure.set(1));

    let error = StorageMut::make_unique(&mut storage)
        .expect_err("injected COW copy failure must remain observable");
    assert!(matches!(
        error,
        HephaestusBackendError::Device {
            operation: "storage uniqueness copy",
            source: HephaestusError::DispatchFailed { message }
        } if message == "injected device copy failure"
    ));
    assert_eq!(storage.allocation_id(), allocation_before);
    assert_eq!(retained.allocation_id(), allocation_before);

    let mut actual = [0; 4];
    device
        .download(storage.buffer(), &mut actual)
        .expect("invariant: retained allocation remains readable");
    assert_eq!(actual, [3, 5, 7, 11]);
}
