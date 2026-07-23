use super::super::elementwise;
use super::super::CpuBackend;
use crate::backend_ops::ops::{BinaryOp, UnaryOp};
use crate::backend_ops::traits::ElementwiseOps;
use coeus_core::{CpuAddressableStorageMut, Layout, Scalar};

#[allow(clippy::too_many_arguments)]
impl<T: Scalar + leto_ops::Scalar, B: CpuBackend> ElementwiseOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn elementwise_binary(
        &self,
        op: BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        elementwise::elementwise_binary(self, op, a, a_layout, b, b_layout, c, c_layout);
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        elementwise::elementwise_unary(self, op, a, a_layout, c, c_layout);
    }
}
