use super::super::elementwise;
use super::super::CpuBackend;
use crate::backend_ops::ops::{BinaryOp, UnaryOp};
use crate::backend_ops::traits::{ElementwiseOps, ScalarPowerOps};
use coeus_core::{CpuAddressableStorageMut, Float, Layout, Scalar};

#[allow(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error> {
        elementwise::elementwise_binary(self, op, a, a_layout, b, b_layout, c, c_layout)
    }

    #[inline]
    fn elementwise_binary_assign(
        &self,
        op: BinaryOp,
        a: &mut Self::DeviceBuffer<T>,
        a_layout: &mut Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
    ) -> Result<(), Self::Error> {
        elementwise::elementwise_binary_assign(self, op, a, a_layout, b, b_layout)
    }

    #[inline]
    fn elementwise_binary_update(
        &self,
        op: BinaryOp,
        destination: &mut Self::DeviceBuffer<T>,
        destination_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
    ) -> Result<(), Self::Error> {
        elementwise::elementwise_binary_assign(
            self,
            op,
            destination,
            destination_layout,
            rhs,
            rhs_layout,
        )
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        elementwise::elementwise_unary(self, op, a, a_layout, c, c_layout)
    }
}

impl<T: Float + leto_ops::Scalar + leto_ops::RealScalar, B: CpuBackend> ScalarPowerOps<T> for B
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    #[inline]
    fn elementwise_pow_scalar(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        exponent: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        elementwise::elementwise_pow_scalar(
            self,
            exponent,
            input,
            input_layout,
            output,
            output_layout,
        )
    }
}
