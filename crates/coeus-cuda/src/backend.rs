use crate::CudaBackendError;
use coeus_core::{Backend, BackendError, ComputeBackend, Scalar, Storage, StorageMut};
use hephaestus_core::CommandStream;
use hephaestus_cuda::{ComputeDevice, CudaDevice, KernelDevice};
use std::sync::OnceLock;

pub mod ops;

/// Scalar types supported by the CUDA backend and Hephaestus fusion.
pub trait CudaScalar: Scalar + leto_ops::Scalar + hephaestus_cuda::CudaFusionScalar {}

impl CudaScalar for f32 {}
impl CudaScalar for f64 {}
impl CudaScalar for eunomia::F16 {}
impl CudaScalar for eunomia::Bf16 {}
impl CudaScalar for i32 {}

static CUDA_DEVICE: OnceLock<CudaDevice> = OnceLock::new();

/// Retrieve the process-global CUDA device, reporting acquisition failures.
///
/// # Errors
///
/// Returns the typed Hephaestus acquisition failure when CUDA is unavailable.
pub fn try_get_cuda_device() -> hephaestus_core::Result<&'static CudaDevice> {
    if let Some(device) = CUDA_DEVICE.get() {
        return Ok(device);
    }
    let candidate = CudaDevice::try_default()?;
    let _ = CUDA_DEVICE.set(candidate);
    CUDA_DEVICE
        .get()
        .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
            message: "CUDA device initialization did not publish the acquired device".to_owned(),
        })
}

/// NVIDIA CUDA acceleration backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBackend;

impl CudaBackend {
    /// Construct a new CUDA backend instance.
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: provider storage never exposes host slices, and successful fill and
// transfer operations initialize their complete destination buffers.
unsafe impl ComputeBackend for CudaBackend {
    type Error = crate::CudaBackendError;
    type DeviceBuffer<T: Scalar> = coeus_accelerator::HephaestusStorage<crate::CudaBackend, T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "cuda"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn host_staging_crossover(&self) -> usize {
        // Measured 2026-10-15 (Ultra 9 285K, RTX 5080, driver 610.60;
        // `ops_bench -- "Reduction crossover"`, f32 scalar sum, criterion
        // means, hephaestus with the scalar axis fast-path): host staging
        // wins at n<=65536 (~91-99us vs ~108-136us device) and the device
        // wins at n>=262144 (~117-119us vs ~525-590us staged). 131072 is
        // the 2^17 midpoint of the measured bracket.
        131_072
    }

    #[inline]
    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // SAFETY: this backend forwards the caller's initialization
        // obligation to the shared provider backend.
        unsafe { coeus_accelerator::HephaestusBackend::<CudaBackend>::new().allocate(len) }
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        coeus_accelerator::HephaestusStorage::<CudaBackend, _>::new(len)
            .map_err(|source| CudaBackendError::dispatch("allocate_zeroed", source))
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error> {
        let size = dst.len();
        if size == 0 {
            return Ok(());
        }

        if val.has_zero_bit_pattern() {
            return self.fill_zero(dst);
        }

        let data = vec![val; size];
        self.copy_to_device(&data, dst)
    }

    #[inline]
    fn fill_zero<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>) -> Result<(), Self::Error> {
        let device = try_get_cuda_device()
            .map_err(|source| CudaBackendError::dispatch("fill_zero", source))?;
        dst.make_unique()?;
        let mut stream = device
            .stream()
            .map_err(|source| CudaBackendError::dispatch("fill_zero", source))?;
        stream
            .fill_zero(dst.buffer())
            .map_err(|source| CudaBackendError::dispatch("fill_zero", source))?;
        stream
            .submit()
            .map_err(|source| CudaBackendError::dispatch("fill_zero", source))
    }

    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(BackendError::BufferLengthMismatch {
                operation: "copy_to_device",
                source_len: src.len(),
                destination_len: dst.len(),
            }
            .into());
        }
        let device = try_get_cuda_device()
            .map_err(|source| CudaBackendError::dispatch("copy_to_device", source))?;
        dst.make_unique()?;
        device
            .write_buffer(dst.buffer(), src)
            .map_err(|source| CudaBackendError::dispatch("copy_to_device", source))
    }

    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(BackendError::BufferLengthMismatch {
                operation: "copy_to_host",
                source_len: src.len(),
                destination_len: dst.len(),
            }
            .into());
        }
        let device = try_get_cuda_device()
            .map_err(|source| CudaBackendError::dispatch("copy_to_host", source))?;
        device
            .download(src.buffer(), dst)
            .map_err(|source| CudaBackendError::dispatch("copy_to_host", source))
    }
}

// SAFETY: CPU fallback dispatch joins every invocation before returning.
unsafe impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, f);
    }
}
