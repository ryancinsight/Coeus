use super::{rotate_half, RotateHalfProvider};
use crate::{HephaestusBackend, HephaestusBackendError, HephaestusStorage};
use coeus_core::{Layout, Scalar};

impl<P, T> coeus_ops::RotateHalfOps<T> for HephaestusBackend<P>
where
    P: RotateHalfProvider<T>,
    T: Scalar,
    hephaestus_core::IdentityOp: hephaestus_core::UnaryExpr<
        <P::Operations as hephaestus_core::ElementwiseOps<P::Device, T>>::Dialect,
    >,
    hephaestus_core::NegOp: hephaestus_core::UnaryExpr<
        <P::Operations as hephaestus_core::ElementwiseOps<P::Device, T>>::Dialect,
    >,
{
    fn rotate_half_storage(
        &self,
        input: &Self::DeviceBuffer<T>,
        layout: &Layout,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        rotate_half::<P, T>(input.buffer(), layout)
            .map(HephaestusStorage::from_buffer)
            .map_err(|source| HephaestusBackendError::device("rotate_half", source))
    }
}
