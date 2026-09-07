//! Coeus elementwise contracts implemented by the Hephaestus WGPU provider.

use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, ElementwiseProvider, HephaestusBackend,
    HephaestusStorage, ParameterizedElementwiseProvider, ScalarPowerProvider,
};
use hephaestus_core::DialectScalar;
use hephaestus_wgpu::{WgpuElementwiseOps, WgpuParameterizedUnaryOps, Wgsl};

impl ParameterizedElementwiseProvider for WgpuBackend {
    type Operations = WgpuParameterizedUnaryOps;
}

impl ElementwiseProvider<f32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

impl ElementwiseProvider<i32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ElementwiseProvider<u32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
}

impl<T> coeus_ops::ElementwiseOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl> + bytemuck::Pod,
    WgpuBackend: ElementwiseProvider<T>,
{
    #[inline]
    fn elementwise_binary(
        &self,
        operation: coeus_ops::BinaryOp,
        lhs: &Self::DeviceBuffer<T>,
        lhs_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let lhs = HephaestusStorage::<WgpuBackend, T>::from_arc(lhs.buffer.clone());
        let rhs = HephaestusStorage::<WgpuBackend, T>::from_arc(rhs.buffer.clone());
        let mut output = HephaestusStorage::<WgpuBackend, T>::from_arc(output.buffer.clone());
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_binary(
                operation,
                &lhs,
                lhs_layout,
                &rhs,
                rhs_layout,
                &mut output,
                output_layout,
            )
            .map_err(Into::into)
    }

    #[inline]
    fn elementwise_unary(
        &self,
        operation: coeus_ops::UnaryOp,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let input = HephaestusStorage::<WgpuBackend, T>::from_arc(input.buffer.clone());
        let mut output = HephaestusStorage::<WgpuBackend, T>::from_arc(output.buffer.clone());
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_unary(operation, &input, input_layout, &mut output, output_layout)
            .map_err(Into::into)
    }
}

impl coeus_ops::ScalarPowerOps<f32> for WgpuBackend
where
    WgpuBackend: ScalarPowerProvider<f32>,
{
    #[inline]
    fn elementwise_pow_scalar(
        &self,
        input: &Self::DeviceBuffer<f32>,
        input_layout: &Layout,
        exponent: f32,
        output: &mut Self::DeviceBuffer<f32>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let input = HephaestusStorage::<WgpuBackend, f32>::from_arc(input.buffer.clone());
        let mut output = HephaestusStorage::<WgpuBackend, f32>::from_arc(output.buffer.clone());
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_pow_scalar(&input, input_layout, exponent, &mut output, output_layout)
            .map_err(Into::into)
    }
}
