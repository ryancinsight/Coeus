//! Injectable test device, buffer, and providers for storage tests.

use super::super::*;
use crate::reduction::HephaestusProvider;
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use std::{
    cell::Cell,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

std::thread_local! {
    pub(super) static DOWNLOADS: Cell<usize> = const { Cell::new(0) };
    pub(super) static DEVICE_COPIES: Cell<usize> = const { Cell::new(0) };
    pub(super) static FAIL_UNINITIALIZED_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    pub(super) static FAIL_DEVICE_COPIES: Cell<usize> = const { Cell::new(0) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TestInitialization {
    Zeroed,
    Uninitialized,
}

#[derive(Debug, Clone)]
pub(super) struct TestBuffer<T: eunomia::Pod> {
    pub(super) bytes: Arc<Mutex<Vec<u8>>>,
    pub(super) len: usize,
    pub(super) tier: MemoryTier,
    pub(super) initialization: TestInitialization,
    pub(super) marker: PhantomData<T>,
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
pub(super) struct TestProvider;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct UnavailableProvider;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct TestDevice;

pub(super) fn byte_len<T: eunomia::Pod>(len: usize) -> hephaestus_core::Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| HephaestusError::AllocationFailed {
            message: "test buffer size overflow".to_owned(),
        })
}

pub(super) fn empty_buffer<T: eunomia::Pod>(
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

pub(super) fn require_len<T: eunomia::Pod>(
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

pub(super) fn assert_device_unavailable(
    error: crate::HephaestusBackendError,
    expected_operation: &str,
) {
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

pub(super) fn expect_backend_failure<T>(
    result: Result<T, crate::HephaestusBackendError>,
    expectation: &str,
) -> crate::HephaestusBackendError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("{expectation}"),
    }
}
