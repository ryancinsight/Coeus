use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;
use coeus_hephaestus::{matmul, MatmulBackend};
use hephaestus_core::DenseProductOps;
use hephaestus_wgpu::{DialectScalar, WgpuDenseProductOps, WgpuDevice, Wgsl};

impl<T> MatmulBackend<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuDenseProductOps: DenseProductOps<WgpuDevice, T>,
{
    type Operations = WgpuDenseProductOps;
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl<T> coeus_ops::MatmulOps<T> for WgpuBackend
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
