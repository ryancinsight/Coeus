use crate::{
    backend::{CudaBackend, CudaScalar},
    CudaBackendError,
};
use coeus_core::Layout;
use coeus_hephaestus::{rotate_half, RotateHalfProvider};
use hephaestus_core::{ElementwiseOps, IdentityOp, NegOp, UnaryExpr};
use hephaestus_cuda::{CudaDevice, CudaElementwiseOps};

impl<T> RotateHalfProvider<T> for CudaBackend
where
    T: CudaScalar,
    CudaElementwiseOps: ElementwiseOps<CudaDevice, T>,
{
    type Operations = CudaElementwiseOps;
}

impl<T> coeus_ops::RotateHalfOps<T> for CudaBackend
where
    T: CudaScalar,
    CudaElementwiseOps: ElementwiseOps<CudaDevice, T>,
    IdentityOp: UnaryExpr<<CudaElementwiseOps as ElementwiseOps<CudaDevice, T>>::Dialect>,
    NegOp: UnaryExpr<<CudaElementwiseOps as ElementwiseOps<CudaDevice, T>>::Dialect>,
{
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<T>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        rotate_half::<Self, _>(input.buffer(), layout)
            .map(coeus_hephaestus::HephaestusStorage::from_buffer)
            .map_err(|source| CudaBackendError::dispatch("rotate_half", source))
    }
}
