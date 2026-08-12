use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackend, HephaestusStorage, ReductionProvider};
use hephaestus_cuda::{
    CudaAxisReductionOps, CudaC, CudaScanOps, CumProdOp, CumSumOp, DialectScalar, IdentityToken,
    MaxOp, MinOp, OpIdentity, ProdOp, SumOp,
};

impl<T> ReductionProvider<T> for CudaBackend
where
    T: CudaScalar
        + DialectScalar<CudaC>
        + bytemuck::Pod
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, CudaC>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, CudaC>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, CudaC>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, CudaC>
        + OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, CudaC>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, CudaC>,
{
    type AxisOperations = CudaAxisReductionOps;
    type ScanOperations = CudaScanOps;
}

impl<T> coeus_ops::ReductionOps<T> for CudaBackend
where
    T: CudaScalar + DialectScalar<CudaC> + bytemuck::Pod,
    CudaBackend: ReductionProvider<T>,
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .reduce(op, &input, a_layout, axis, &mut output, c_layout)
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .cumsum(&input, a_layout, axis, &mut output, c_layout)
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .suffix_sum(&input, a_layout, axis, &mut output, c_layout)
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .cumprod(&input, a_layout, axis, &mut output, c_layout)
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .suffix_prod(&input, a_layout, axis, &mut output, c_layout)
            .map_err(Into::into)
    }
}
