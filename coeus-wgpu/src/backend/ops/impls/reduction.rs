use super::super::reduction;
use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;

impl<T: WgpuScalar + leto_ops::Scalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>>
    coeus_ops::ReductionOps<T> for WgpuBackend
{
    #[inline]
    fn reduce(
        &self,
        op: coeus_ops::ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        reduction::dispatch_reduce(op, a, a_layout, axis, c, c_layout);
    }
}
