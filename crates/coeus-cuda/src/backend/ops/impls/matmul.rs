use crate::backend::{get_cuda_device, CudaBackend, CudaScalar};
use coeus_core::Layout;
use coeus_hephaestus::{matmul, MatmulBackend};
use hephaestus_core::{ComputeDevice, DenseProductOps, HephaestusError};
use hephaestus_cuda::{CudaDenseProductOps, CudaDevice};

impl<T> MatmulBackend<T> for CudaBackend
where
    T: CudaScalar,
    CudaDenseProductOps: DenseProductOps<CudaDevice, T>,
{
    type Device = CudaDevice;
    type Operations = CudaDenseProductOps;

    fn matmul_device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn matmul_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn matmul_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        crate::CudaBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::MatmulOps<T> for CudaBackend
where
    T: CudaScalar,
    CudaDenseProductOps: DenseProductOps<CudaDevice, T>,
{
    #[inline]
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        matmul::<Self, T>(a, a_layout, b, b_layout, c, c_layout)
    }
}
