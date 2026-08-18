use super::{dispatch, provider::MatmulProvider};
use crate::HephaestusBackend;
use coeus_core::{Layout, Scalar};

impl<P, T> coeus_ops::MatmulOps<T> for HephaestusBackend<P>
where
    P: MatmulProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::matmul::<HephaestusBackend<P>, T>(a, a_layout, b, b_layout, c, c_layout)
    }
}
