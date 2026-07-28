use crate::storage::CudaStorage;
use coeus_core::{Backend, ComputeBackend, Scalar, Storage};
use coeus_hephaestus::SharedHephaestusError;
use hephaestus_cuda::{ComputeDevice, CudaDevice};
use std::sync::OnceLock;

pub mod ops;

/// Trait combining [`Scalar`] with the CUDA type-name mapping required for kernel codegen.
pub trait CudaScalar: Scalar + leto_ops::Scalar {
    /// CUDA C type name string used in NVRTC-compiled kernel source.
    const CUDA_TYPE: &'static str;
}

impl CudaScalar for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaScalar for f64 {
    const CUDA_TYPE: &'static str = "double";
}

impl CudaScalar for eunomia::F16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaScalar for eunomia::Bf16 {
    const CUDA_TYPE: &'static str = "__nv_bfloat16";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}

static CUDA_DEVICE: OnceLock<Result<CudaDevice, SharedHephaestusError>> = OnceLock::new();

/// Retrieve a reference to the global lazily-initialized hephaestus CUDA device.
pub fn get_cuda_device() -> Result<&'static CudaDevice, crate::CudaBackendError> {
    CUDA_DEVICE
        .get_or_init(|| CudaDevice::try_default().map_err(SharedHephaestusError::new))
        .as_ref()
        .map_err(|source| crate::CudaBackendError::initialization(source.clone()))
}

/// NVIDIA CUDA acceleration backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBackend;

impl coeus_core::backend::private::Sealed for CudaBackend {}

impl CudaBackend {
    /// Construct a new CUDA backend instance.
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CudaBackend {
    type Error = crate::CudaBackendError;
    type DeviceBuffer<T: Scalar> = CudaStorage<T>;
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
    fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        CudaStorage::try_new(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) -> Result<(), Self::Error> {
        let size = dst.len();
        if size == 0 {
            return Ok(());
        }
        let bytes = size.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            CudaBackendError::from(coeus_core::BackendError::Overflow {
                operation: "fill",
                reason: "element-count to byte-size arithmetic overflow",
            })
        })?;
        let device = get_cuda_device()?;
        device
            .bind()
            .map_err(|source| CudaBackendError::provider("fill", source))?;

        // Fast path: if the value is bitwise zero, use cuMemsetD8_v2
        let is_zero = Scalar::to_f64(val) == 0.0;
        if is_zero {
            unsafe {
                let res = cuda_core::sys::cuMemsetD8_v2(dst.cu_deviceptr(), 0, bytes);
                if res == 0 {
                    return Ok(());
                }
            }
        }

        // If T is 32-bit, use cuMemsetD32_v2
        if std::mem::size_of::<T>() == 4 {
            let val_u32 = unsafe {
                let mut tmp = 0u32;
                std::ptr::copy_nonoverlapping(
                    &val as *const T as *const u8,
                    &mut tmp as *mut u32 as *mut u8,
                    4,
                );
                tmp
            };
            unsafe {
                let res = cuda_core::sys::cuMemsetD32_v2(dst.cu_deviceptr(), val_u32, size);
                if res == 0 {
                    return Ok(());
                }
            }
        }

        // If T is 16-bit, use cuMemsetD16_v2
        if std::mem::size_of::<T>() == 2 {
            let val_u16 = unsafe {
                let mut tmp = 0u16;
                std::ptr::copy_nonoverlapping(
                    &val as *const T as *const u8,
                    &mut tmp as *mut u16 as *mut u8,
                    2,
                );
                tmp
            };
            unsafe {
                let res = cuda_core::sys::cuMemsetD16_v2(dst.cu_deviceptr(), val_u16, size);
                if res == 0 {
                    return Ok(());
                }
            }
        }

        let data = vec![val; size];
        self.copy_to_device(&data, dst)
    }

    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        let device = get_cuda_device()?;
        device
            .write_buffer(&dst.buffer, src)
            .map_err(|source| CudaBackendError::provider("copy_to_device", source))
    }

    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        let device = get_cuda_device()?;
        device
            .download(&src.buffer, dst)
            .map_err(|source| CudaBackendError::provider("copy_to_host", source))
    }
}

impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, f);
    }
}
