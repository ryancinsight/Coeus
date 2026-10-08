//! Coeus elementwise contracts implemented by the Hephaestus WGPU provider.

use crate::backend::WgpuBackend;
use coeus_core::{Float, Layout, Scalar};
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, ElementwiseProvider, HephaestusBackend,
    ParameterizedElementwiseProvider, ScalarPowerProvider,
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

impl ElementwiseProvider<f64> for WgpuBackend {
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

// Per-type providers (mirroring CUDA): the `ScalarPowerDispatch` blanket
// impl's dialect projections only normalize for concrete `T`, so a generic
// provider impl cannot satisfy the associated-type bound.
//
// f64 is deliberately absent: the f64 `pow(lhs, rhs)` shader passes naga
// validation yet crashes native backends (access violation at dispatch —
// f64 buffers, upload, and f64 arithmetic shaders are all proven working
// by the rotate-half f64 test, isolating the `pow` builtin codegen). Until
// hephaestus fixes f64 `pow` codegen, admitting f64 here would arm a
// process-crashing trap; the missing impl rejects it at compile time
// instead. `WgpuScalar` is deliberately not required since scalar power
// runs on the plain elementwise seam.
impl ScalarPowerProvider<f32> for WgpuBackend {
    type Operations = WgpuElementwiseOps;
}

impl<T> coeus_ops::ElementwiseOps<T> for WgpuBackend
where
    T: Scalar + DialectScalar<Wgsl> + bytemuck::Pod,
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
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_binary(
                operation,
                lhs,
                lhs_layout,
                rhs,
                rhs_layout,
                output,
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
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_unary(operation, input, input_layout, output, output_layout)
            .map_err(Into::into)
    }
}

impl<T> coeus_ops::ScalarPowerOps<T> for WgpuBackend
where
    T: Float + DialectScalar<Wgsl> + bytemuck::Pod,
    WgpuBackend: ScalarPowerProvider<T>,
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
        HephaestusBackend::<WgpuBackend>::new()
            .elementwise_pow_scalar(input, input_layout, exponent, output, output_layout)
            .map_err(Into::into)
    }
}
