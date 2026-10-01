use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::{Float, Layout};
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, ElementwiseProvider, HephaestusBackend,
    ParameterizedElementwiseProvider, ScalarPowerProvider,
};
use hephaestus_cuda::{CudaC, CudaElementwiseOps, CudaParameterizedUnaryOps, DialectScalar};

impl ParameterizedElementwiseProvider for CudaBackend {
    type Operations = CudaParameterizedUnaryOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<f32> for CudaBackend {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<f64> for CudaBackend {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ElementwiseProvider<i32> for CudaBackend {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ScalarPowerProvider<f32> for CudaBackend {
    type Operations = CudaElementwiseOps;
}

// SAFETY: Provider kernels overwrite every logical output on success without reading prior contents.
unsafe impl ScalarPowerProvider<f64> for CudaBackend {
    type Operations = CudaElementwiseOps;
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl<T> coeus_ops::ElementwiseOps<T> for CudaBackend
where
    T: CudaScalar + DialectScalar<CudaC> + bytemuck::Pod,
    CudaBackend: ElementwiseProvider<T>,
{
    #[inline]
    fn elementwise_binary(
        &self,
        op: coeus_ops::BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        HephaestusBackend::<CudaBackend>::new()
            .elementwise_binary(op, a, a_layout, b, b_layout, c, c_layout)
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: coeus_ops::UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        HephaestusBackend::<CudaBackend>::new().elementwise_unary(op, a, a_layout, c, c_layout)
    }
}

// SAFETY: Overwrite methods initialize every logical output on success; accumulation methods require initialized outputs.
unsafe impl<T> coeus_ops::ScalarPowerOps<T> for CudaBackend
where
    T: Float + CudaScalar + DialectScalar<CudaC> + bytemuck::Pod,
    CudaBackend: ScalarPowerProvider<T>,
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
        HephaestusBackend::<CudaBackend>::new().elementwise_pow_scalar(
            input,
            input_layout,
            exponent,
            output,
            output_layout,
        )
    }
}
