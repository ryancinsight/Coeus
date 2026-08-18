use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError, WgpuScalar};
use coeus_core::Layout;
use coeus_hephaestus::{matmul, MatmulBackend};
use hephaestus_core::{ComputeDevice, DenseProductOps, HephaestusError};
use hephaestus_wgpu::{DialectScalar, WgpuDenseProductOps, WgpuDevice, Wgsl};

impl<T> MatmulBackend<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuDenseProductOps: DenseProductOps<WgpuDevice, T>,
{
    type Device = WgpuDevice;
    type Operations = WgpuDenseProductOps;

    fn matmul_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn matmul_buffer(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<T> {
        storage.buffer.as_ref()
    }

    fn matmul_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl<T> coeus_ops::MatmulOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuDenseProductOps: DenseProductOps<WgpuDevice, T>,
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
