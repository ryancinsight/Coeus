use crate::storage::CudaStorage;
use coeus_core::{Backend, ComputeBackend, Scalar, Storage};
use hephaestus_core::{CommandStream, KernelDevice};
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

static CUDA_DEVICE: OnceLock<CudaDevice> = OnceLock::new();

/// Retrieve a reference to the global lazily-initialized hephaestus CUDA device.
pub fn get_cuda_device() -> &'static CudaDevice {
    CUDA_DEVICE.get_or_init(|| {
        CudaDevice::try_default().expect("Failed to initialize hephaestus-cuda device")
    })
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
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CudaStorage::uninitialized(len)
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CudaStorage::new(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let size = dst.len();
        if size == 0 {
            return;
        }

        if val.has_zero_bit_pattern() {
            self.fill_zero(dst);
            return;
        }

        let device = get_cuda_device();
        device.bind().expect("fill: failed to bind CUDA device");

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
                    return;
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
                    return;
                }
            }
        }

        let data = vec![val; size];
        self.copy_to_device(&data, dst);
    }

    #[inline]
    fn fill_zero<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>) {
        let device = get_cuda_device();
        let mut stream = device
            .stream()
            .expect("CUDA zero fill stream creation failed");
        stream
            .fill_zero(dst.buffer.as_ref())
            .expect("CUDA zero fill encoding failed");
        stream.submit().expect("CUDA zero fill submission failed");
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        let device = get_cuda_device();
        device
            .write_buffer(&dst.buffer, src)
            .expect("copy_to_device: write_buffer failed");
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let device = get_cuda_device();
        device
            .download(&src.buffer, dst)
            .expect("copy_to_host: download failed");
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
