use std::sync::Mutex;

use coeus_core::Storage;
use coeus_hephaestus::{HephaestusBackendError, HephaestusProvider, HephaestusStorage};
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use themis::{MemoryTier, PlacementHint};

pub(crate) struct TestBuffer<T> {
    pub(crate) values: Mutex<Vec<T>>,
    tier: MemoryTier,
}

pub(crate) fn tier(hint: PlacementHint) -> MemoryTier {
    match hint {
        PlacementHint::Tier(tier) => tier,
        PlacementHint::Current
        | PlacementHint::Numa(_)
        | PlacementHint::Domain(_)
        | PlacementHint::Any => MemoryTier::Dram,
    }
}

impl<T> TestBuffer<T> {
    fn new(values: Vec<T>, tier: MemoryTier) -> Self {
        Self {
            values: Mutex::new(values),
            tier,
        }
    }
}

impl<T> DeviceBuffer<T> for TestBuffer<T> {
    fn len(&self) -> usize {
        self.values.lock().expect("test buffer lock").len()
    }

    fn tier(&self) -> MemoryTier {
        self.tier
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct TestDevice;

pub(crate) fn length_mismatch<T>(host_len: usize, buffer: &TestBuffer<T>) -> HephaestusError {
    HephaestusError::LengthMismatch {
        host_len,
        device_len: buffer.len(),
    }
}

impl ComputeDevice for TestDevice {
    type Buffer<T: eunomia::Pod> = TestBuffer<T>;

    fn backend_name(&self) -> &'static str {
        "attention-bridge-test"
    }

    fn topology(&self) -> Option<&themis::GpuTopology> {
        None
    }

    fn alloc_zeroed_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(vec![T::zeroed(); len], tier(hint)))
    }

    fn alloc_uninitialized_with_hint<T: eunomia::Pod>(
        &self,
        len: usize,
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        self.alloc_zeroed_with_hint(len, hint)
    }

    fn upload_with_hint<T: eunomia::Pod>(
        &self,
        host: &[T],
        hint: PlacementHint,
    ) -> hephaestus_core::Result<Self::Buffer<T>> {
        Ok(TestBuffer::new(host.to_vec(), tier(hint)))
    }

    fn download<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> hephaestus_core::Result<()> {
        let values = buffer.values.lock().expect("test buffer lock");
        if values.len() != out.len() {
            return Err(length_mismatch(out.len(), buffer));
        }
        out.copy_from_slice(&values);
        Ok(())
    }

    fn write_buffer<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        host: &[T],
    ) -> hephaestus_core::Result<()> {
        let mut values = buffer.values.lock().expect("test buffer lock");
        if values.len() != host.len() {
            return Err(length_mismatch(host.len(), buffer));
        }
        values.copy_from_slice(host);
        Ok(())
    }

    fn write_sub_buffer<T: eunomia::Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        offset: usize,
        host: &[T],
    ) -> hephaestus_core::Result<()> {
        let mut values = buffer.values.lock().expect("test buffer lock");
        let device_len = values.len();
        let end =
            offset
                .checked_add(host.len())
                .ok_or_else(|| HephaestusError::TransferFailed {
                    message: "test sub-buffer range overflow".into(),
                })?;
        let destination = values
            .get_mut(offset..end)
            .ok_or(HephaestusError::LengthMismatch {
                host_len: end,
                device_len,
            })?;
        destination.copy_from_slice(host);
        Ok(())
    }

    fn copy_buffer<T: eunomia::Pod>(
        &self,
        src: &Self::Buffer<T>,
        dst: &Self::Buffer<T>,
    ) -> hephaestus_core::Result<()> {
        let source = src.values.lock().expect("test buffer lock").clone();
        self.write_buffer(dst, &source)
    }

    fn synchronize(&self) -> hephaestus_core::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct TestProvider;

pub(crate) static DEVICE: TestDevice = TestDevice;

// SAFETY: test buffers own synchronized host memory and remain valid for the
// lifetime of every retained handle; dispatch is synchronous.
unsafe impl HephaestusProvider for TestProvider {
    type Device = TestDevice;
    type Error = HephaestusBackendError;

    const NAME: &'static str = "attention-bridge-test";

    fn device() -> &'static Self::Device {
        &DEVICE
    }

    fn try_device() -> hephaestus_core::Result<&'static Self::Device> {
        Ok(Self::device())
    }
}

pub(crate) fn storage(len: usize) -> HephaestusStorage<TestProvider, f32> {
    HephaestusStorage::new(len).expect("invariant: test device allocation succeeds")
}

pub(crate) fn write_storage(storage: &HephaestusStorage<TestProvider, f32>, values: &[f32]) {
    DEVICE
        .write_buffer(storage.buffer(), values)
        .expect("invariant: test storage and input lengths match");
}

pub(crate) fn read_storage(storage: &HephaestusStorage<TestProvider, f32>) -> Vec<f32> {
    let mut values = vec![0.0; storage.len()];
    DEVICE
        .download(storage.buffer(), &mut values)
        .expect("invariant: test storage and output lengths match");
    values
}
