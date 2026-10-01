//! Coeus elementwise contracts implemented by the Hephaestus WGPU provider.

use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::Layout;
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, ElementwiseProvider, HephaestusBackend,
    ParameterizedElementwiseProvider, ScalarPowerProvider,
};
use hephaestus_core::DialectScalar;
use hephaestus_wgpu::{WgpuElementwiseOps, WgpuParameterizedUnaryOps, Wgsl};

impl ParameterizedElementwiseProvider for WgpuBackend {
    type Operations = WgpuParameterizedUnaryOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<f32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<i32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<u32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ScalarPowerProvider<f32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl<T> coeus_ops::ElementwiseOps<T> for WgpuBackend
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
        HephaestusBackend::<WgpuBackend>::new().elementwise_binary(
            operation,
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output,
            output_layout,
        )
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
        HephaestusBackend::<WgpuBackend>::new().elementwise_unary(
            operation,
            input,
            input_layout,
            output,
            output_layout,
        )
    }
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl coeus_ops::ScalarPowerOps<f32> for WgpuBackend
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
        HephaestusBackend::<WgpuBackend>::new().elementwise_pow_scalar(
            input,
            input_layout,
            exponent,
            output,
            output_layout,
        )
    }
}
