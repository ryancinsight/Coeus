use super::super::matmul;
use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;

impl<T: WgpuScalar + leto_ops::Scalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>>
    coeus_ops::MatmulOps<T> for WgpuBackend
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
        matmul::dispatch_matmul(a, a_layout, b, b_layout, c, c_layout)
    }
}
