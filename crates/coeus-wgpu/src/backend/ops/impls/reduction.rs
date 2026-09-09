use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackend, ReductionProvider};
use hephaestus_core::{
    CumProdOp, CumSumOp, DialectScalar, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp, SumOp,
};
use hephaestus_wgpu::{WgpuAxisReductionOps, WgpuScanOps, Wgsl};

impl<T> ReductionProvider<T> for WgpuBackend
where
    T: WgpuScalar
        + leto_ops::Scalar
        + DialectScalar<Wgsl>
        + bytemuck::Pod
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, Wgsl>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Wgsl>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, Wgsl>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>
        + OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, Wgsl>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, Wgsl>,
{
    type AxisOperations = WgpuAxisReductionOps;
    type ScanOperations = WgpuScanOps;
}

impl<T> coeus_ops::ReductionOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl> + bytemuck::Pod,
    WgpuBackend: ReductionProvider<T>,
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
    ) -> Result<(), Self::Error> {
        HephaestusBackend::<WgpuBackend>::new()
            .reduce(op, a, a_layout, axis, c, c_layout)
            .map_err(Into::into)
    }

    #[inline]
    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        HephaestusBackend::<WgpuBackend>::new()
            .cumsum(a, a_layout, axis, c, c_layout)
            .map_err(Into::into)
    }

    #[inline]
    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        HephaestusBackend::<WgpuBackend>::new()
            .suffix_sum(a, a_layout, axis, c, c_layout)
            .map_err(Into::into)
    }

    #[inline]
    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        HephaestusBackend::<WgpuBackend>::new()
            .cumprod(a, a_layout, axis, c, c_layout)
            .map_err(Into::into)
    }

    #[inline]
    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        HephaestusBackend::<WgpuBackend>::new()
            .suffix_prod(a, a_layout, axis, c, c_layout)
            .map_err(Into::into)
    }
}
