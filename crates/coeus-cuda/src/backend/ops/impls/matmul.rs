use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Layout;

impl<T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>> coeus_ops::MatmulOps<T>
    for CudaBackend
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
        self.cuda_matmul(a, a_layout, b, b_layout, c, c_layout)
    }
}
