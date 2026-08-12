use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::{Float, Layout};
use coeus_hephaestus::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, ElementwiseProvider, HephaestusBackend,
    HephaestusStorage, ParameterizedElementwiseProvider, ScalarPowerProvider,
};
use hephaestus_cuda::{CudaC, CudaElementwiseOps, CudaParameterizedUnaryOps, DialectScalar};

impl ParameterizedElementwiseProvider for CudaBackend {
    type Operations = CudaParameterizedUnaryOps;
}

impl ElementwiseProvider<f32> for CudaBackend {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ActivationUnaryOperations;
}

// `f64` elementwise is not wired here: the bridge's `BinaryElementwiseDispatch`
// contract requires the six comparison `TypedBinaryExpr<CudaC, T>` operations,
// which hephaestus-cuda implements for `f32`/`u32`/`i32` but not `f64` at the
// pinned gitlink. `f64` scalar power remains available through
// `ScalarPowerProvider<f64>` below; restoring `f64` elementwise is a
// hephaestus-cuda capability follow-up (recorded in the SUBSTRATE-002 ledger).

impl ElementwiseProvider<i32> for CudaBackend {
    type Operations = CudaElementwiseOps;
    type UnaryOperations = ArithmeticUnaryOperations;
}

impl ScalarPowerProvider<f32> for CudaBackend {
    type Operations = CudaElementwiseOps;
}

impl ScalarPowerProvider<f64> for CudaBackend {
    type Operations = CudaElementwiseOps;
}

impl<T> coeus_ops::ElementwiseOps<T> for CudaBackend
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
        let lhs = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let rhs = HephaestusStorage::<CudaBackend, T>::from_arc(b.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .elementwise_binary(op, &lhs, a_layout, &rhs, b_layout, &mut output, c_layout)
            .map_err(Into::into)
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(a.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(c.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .elementwise_unary(op, &input, a_layout, &mut output, c_layout)
            .map_err(Into::into)
    }
}

impl<T> coeus_ops::ScalarPowerOps<T> for CudaBackend
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
        let input = HephaestusStorage::<CudaBackend, T>::from_arc(input.buffer.clone());
        let mut output = HephaestusStorage::<CudaBackend, T>::from_arc(output.buffer.clone());
        HephaestusBackend::<CudaBackend>::new()
            .elementwise_pow_scalar(&input, input_layout, exponent, &mut output, output_layout)
            .map_err(Into::into)
    }
}
