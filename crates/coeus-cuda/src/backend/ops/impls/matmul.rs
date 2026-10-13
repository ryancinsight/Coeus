use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Layout;
use coeus_hephaestus::{matmul, MatmulBackend};
use hephaestus_core::DenseProductOps;
use hephaestus_cuda::{CudaDenseProductOps, CudaDevice};

impl<T> MatmulBackend<T> for CudaBackend
where
    T: CudaScalar,
    CudaDenseProductOps: DenseProductOps<CudaDevice, T>,
{
    type Operations = CudaDenseProductOps;
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl<T> coeus_ops::MatmulOps<T> for CudaBackend
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
