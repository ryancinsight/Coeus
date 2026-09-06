//! Provider-neutral operation dispatch.

use super::provider::{parameterized_unary, ParameterizedElementwiseProvider};
use crate::reduction::{HephaestusProvider, RankedOperand};
use coeus_ops::{BinaryOp, UnaryOp};
use hephaestus_core::{
    BinaryExpr, ComputeDevice, DialectScalar, ElementwiseOps as HephaestusElementwiseOps,
    HardtanhGradOp, HardtanhOp, ParameterizedUnaryExpr, ParameterizedUnaryOps, StridedView,
    ThresholdGradOp, ThresholdOp, TypedBinaryExpr, UnaryExpr,
};

fn unsupported_unary_operation(operation: UnaryOp) -> hephaestus_core::HephaestusError {
    hephaestus_core::HephaestusError::DispatchFailed {
        message: format!(
            "unary elementwise operation {operation:?} is not implemented by provider"
        ),
    }
}

/// Provider-neutral binary operation dispatch over a Hephaestus elementwise
/// seam.
pub trait BinaryElementwiseDispatch<D: ComputeDevice, T: eunomia::Pod> {
    /// Execute one Coeus binary operation over ranked strided operands.
    fn binary<const N: usize>(
        device: &D,
        operation: BinaryOp,
        lhs: RankedOperand<'_, D::Buffer<T>, N>,
        rhs: RankedOperand<'_, D::Buffer<T>, N>,
        output: RankedOperand<'_, D::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
}

impl<D, T, E> BinaryElementwiseDispatch<D, T> for E
where
    D: ComputeDevice,
    T: eunomia::Pod + DialectScalar<E::Dialect>,
    E: HephaestusElementwiseOps<D, T> + Default,
    hephaestus_core::AddOp: BinaryExpr<E::Dialect>,
    hephaestus_core::SubOp: BinaryExpr<E::Dialect>,
    hephaestus_core::MulOp: BinaryExpr<E::Dialect>,
    hephaestus_core::DivOp: BinaryExpr<E::Dialect>,
    hephaestus_core::EqOp: TypedBinaryExpr<E::Dialect, T>,
    hephaestus_core::NeOp: TypedBinaryExpr<E::Dialect, T>,
    hephaestus_core::LtOp: TypedBinaryExpr<E::Dialect, T>,
    hephaestus_core::GtOp: TypedBinaryExpr<E::Dialect, T>,
    hephaestus_core::LeOp: TypedBinaryExpr<E::Dialect, T>,
    hephaestus_core::GeOp: TypedBinaryExpr<E::Dialect, T>,
{
    fn binary<const N: usize>(
        device: &D,
        operation: BinaryOp,
        lhs: RankedOperand<'_, D::Buffer<T>, N>,
        rhs: RankedOperand<'_, D::Buffer<T>, N>,
        output: RankedOperand<'_, D::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        let operations = E::default();
        let lhs = StridedView::new(lhs.buffer, lhs.layout);
        let rhs = StridedView::new(rhs.buffer, rhs.layout);
        let output = StridedView::new(output.buffer, output.layout);
        match operation {
            BinaryOp::Add => {
                operations.binary_into::<hephaestus_core::AddOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Sub => {
                operations.binary_into::<hephaestus_core::SubOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Mul => {
                operations.binary_into::<hephaestus_core::MulOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Div => {
                operations.binary_into::<hephaestus_core::DivOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Eq => {
                operations.typed_binary_into::<hephaestus_core::EqOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Ne => {
                operations.typed_binary_into::<hephaestus_core::NeOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Lt => {
                operations.typed_binary_into::<hephaestus_core::LtOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Gt => {
                operations.typed_binary_into::<hephaestus_core::GtOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Le => {
                operations.typed_binary_into::<hephaestus_core::LeOp, N>(device, lhs, rhs, output)
            }
            BinaryOp::Ge => {
                operations.typed_binary_into::<hephaestus_core::GeOp, N>(device, lhs, rhs, output)
            }
        }
    }
}

/// Selects the arithmetic unary operation set supported for integral and
/// floating-point Coeus scalars.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArithmeticUnaryOperations;

/// Selects the activation and arithmetic unary operation set supported for
/// floating-point Coeus scalars.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActivationUnaryOperations;

/// Provider-neutral scalar-power dispatch over a Hephaestus elementwise seam.
pub trait ScalarPowerDispatch<D: ComputeDevice, T: eunomia::Pod> {
    /// Execute `output = input.powf(exponent)` over ranked strided operands.
    fn scalar_power<const N: usize>(
        device: &D,
        input: RankedOperand<'_, D::Buffer<T>, N>,
        exponent: T,
        output: RankedOperand<'_, D::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
}

impl<D, T, E> ScalarPowerDispatch<D, T> for E
where
    D: ComputeDevice,
    T: eunomia::Pod + DialectScalar<E::Dialect>,
    E: HephaestusElementwiseOps<D, T> + Default,
    hephaestus_core::PowOp: BinaryExpr<E::Dialect>,
{
    fn scalar_power<const N: usize>(
        device: &D,
        input: RankedOperand<'_, D::Buffer<T>, N>,
        exponent: T,
        output: RankedOperand<'_, D::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        E::default().scalar_into::<hephaestus_core::PowOp, N>(
            device,
            StridedView::new(input.buffer, input.layout),
            exponent,
            StridedView::new(output.buffer, output.layout),
        )
    }
}

/// Provider-neutral unary operation dispatch over a Hephaestus elementwise
/// seam.
pub trait UnaryElementwiseDispatch<P, T: eunomia::Pod, E>
where
    P: HephaestusProvider,
    E: HephaestusElementwiseOps<P::Device, T>,
{
    /// Execute one Coeus unary operation over ranked strided operands.
    fn unary<const N: usize>(
        device: &P::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
}

impl<P, T, E> UnaryElementwiseDispatch<P, T, E> for ArithmeticUnaryOperations
where
    P: HephaestusProvider,
    T: eunomia::Pod + DialectScalar<E::Dialect>,
    E: HephaestusElementwiseOps<P::Device, T> + Default,
    hephaestus_core::SinOp: UnaryExpr<E::Dialect>,
    hephaestus_core::CosOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ExpOp: UnaryExpr<E::Dialect>,
    hephaestus_core::LnOp: UnaryExpr<E::Dialect>,
    hephaestus_core::NegOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AbsOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SqrtOp: UnaryExpr<E::Dialect>,
    hephaestus_core::RecipOp: UnaryExpr<E::Dialect>,
{
    fn unary<const N: usize>(
        device: &P::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        let operations = E::default();
        let input = StridedView::new(input.buffer, input.layout);
        let output = StridedView::new(output.buffer, output.layout);
        match operation {
            UnaryOp::Sin => {
                operations.unary_into::<hephaestus_core::SinOp, N>(device, input, output)
            }
            UnaryOp::Cos => {
                operations.unary_into::<hephaestus_core::CosOp, N>(device, input, output)
            }
            UnaryOp::Exp => {
                operations.unary_into::<hephaestus_core::ExpOp, N>(device, input, output)
            }
            UnaryOp::Log => {
                operations.unary_into::<hephaestus_core::LnOp, N>(device, input, output)
            }
            UnaryOp::Neg => {
                operations.unary_into::<hephaestus_core::NegOp, N>(device, input, output)
            }
            UnaryOp::Abs => {
                operations.unary_into::<hephaestus_core::AbsOp, N>(device, input, output)
            }
            UnaryOp::Sqrt => {
                operations.unary_into::<hephaestus_core::SqrtOp, N>(device, input, output)
            }
            UnaryOp::Recip => {
                operations.unary_into::<hephaestus_core::RecipOp, N>(device, input, output)
            }
            _ => Err(unsupported_unary_operation(operation)),
        }
    }
}

impl<P, E> UnaryElementwiseDispatch<P, f32, E> for ActivationUnaryOperations
where
    P: HephaestusProvider + ParameterizedElementwiseProvider,
    E: HephaestusElementwiseOps<P::Device, f32> + Default,
    f32: DialectScalar<E::Dialect>,
    hephaestus_core::SinOp: UnaryExpr<E::Dialect>,
    hephaestus_core::CosOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ExpOp: UnaryExpr<E::Dialect>,
    hephaestus_core::LnOp: UnaryExpr<E::Dialect>,
    hephaestus_core::NegOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AbsOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SqrtOp: UnaryExpr<E::Dialect>,
    hephaestus_core::RecipOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ReluOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ReluGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SigmoidOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SigmoidGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::TanhOp: UnaryExpr<E::Dialect>,
    hephaestus_core::TanhGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::GeluOp: UnaryExpr<E::Dialect>,
    hephaestus_core::GeluGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::GeluTanhOp: UnaryExpr<E::Dialect>,
    hephaestus_core::GeluTanhGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SiluOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SiluGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SoftplusOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SoftplusGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::MishOp: UnaryExpr<E::Dialect>,
    hephaestus_core::MishGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::EluOp: UnaryExpr<E::Dialect>,
    hephaestus_core::EluGradOp: UnaryExpr<E::Dialect>,
    hephaestus_core::TanOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AsinOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AcosOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AtanOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SinhOp: UnaryExpr<E::Dialect>,
    hephaestus_core::CoshOp: UnaryExpr<E::Dialect>,
    hephaestus_core::Log2Op: UnaryExpr<E::Dialect>,
    hephaestus_core::Log10Op: UnaryExpr<E::Dialect>,
    hephaestus_core::Exp2Op: UnaryExpr<E::Dialect>,
    hephaestus_core::AtanhOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AsinhOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AcoshOp: UnaryExpr<E::Dialect>,
    hephaestus_core::Expm1Op: UnaryExpr<E::Dialect>,
    hephaestus_core::Log1pOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SignOp: UnaryExpr<E::Dialect>,
    hephaestus_core::FloorOp: UnaryExpr<E::Dialect>,
    hephaestus_core::CeilOp: UnaryExpr<E::Dialect>,
    hephaestus_core::RoundOp: UnaryExpr<E::Dialect>,
    hephaestus_core::TruncOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ErfOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ErfcOp: UnaryExpr<E::Dialect>,
    hephaestus_core::LgammaOp: UnaryExpr<E::Dialect>,
    HardtanhOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    HardtanhGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
{
    fn unary<const N: usize>(
        device: &P::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<()> {
        let input_view = StridedView::new(input.buffer, input.layout);
        let output_view = StridedView::new(output.buffer, output.layout);
        let operations = E::default();
        match operation {
            UnaryOp::Hardtanh(_)
            | UnaryOp::HardtanhGrad(_)
            | UnaryOp::Threshold(_)
            | UnaryOp::ThresholdGrad(_) => parameterized_unary::<P, N>(operation, input, output),
            UnaryOp::Sin => {
                operations.unary_into::<hephaestus_core::SinOp, N>(device, input_view, output_view)
            }
            UnaryOp::Cos => {
                operations.unary_into::<hephaestus_core::CosOp, N>(device, input_view, output_view)
            }
            UnaryOp::Exp => {
                operations.unary_into::<hephaestus_core::ExpOp, N>(device, input_view, output_view)
            }
            UnaryOp::Log => {
                operations.unary_into::<hephaestus_core::LnOp, N>(device, input_view, output_view)
            }
            UnaryOp::Neg => {
                operations.unary_into::<hephaestus_core::NegOp, N>(device, input_view, output_view)
            }
            UnaryOp::Abs => {
                operations.unary_into::<hephaestus_core::AbsOp, N>(device, input_view, output_view)
            }
            UnaryOp::Sqrt => {
                operations.unary_into::<hephaestus_core::SqrtOp, N>(device, input_view, output_view)
            }
            UnaryOp::Recip => operations.unary_into::<hephaestus_core::RecipOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Relu => {
                operations.unary_into::<hephaestus_core::ReluOp, N>(device, input_view, output_view)
            }
            UnaryOp::ReluGrad => operations.unary_into::<hephaestus_core::ReluGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Sigmoid => operations.unary_into::<hephaestus_core::SigmoidOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::SigmoidGrad => operations.unary_into::<hephaestus_core::SigmoidGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Tanh => {
                operations.unary_into::<hephaestus_core::TanhOp, N>(device, input_view, output_view)
            }
            UnaryOp::TanhGrad => operations.unary_into::<hephaestus_core::TanhGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Gelu => {
                operations.unary_into::<hephaestus_core::GeluOp, N>(device, input_view, output_view)
            }
            UnaryOp::GeluGrad => operations.unary_into::<hephaestus_core::GeluGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::GeluTanh => operations.unary_into::<hephaestus_core::GeluTanhOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::GeluTanhGrad => operations.unary_into::<hephaestus_core::GeluTanhGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Silu => {
                operations.unary_into::<hephaestus_core::SiluOp, N>(device, input_view, output_view)
            }
            UnaryOp::SiluGrad => operations.unary_into::<hephaestus_core::SiluGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Softplus => operations.unary_into::<hephaestus_core::SoftplusOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::SoftplusGrad => operations.unary_into::<hephaestus_core::SoftplusGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Mish => {
                operations.unary_into::<hephaestus_core::MishOp, N>(device, input_view, output_view)
            }
            UnaryOp::MishGrad => operations.unary_into::<hephaestus_core::MishGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Elu => {
                operations.unary_into::<hephaestus_core::EluOp, N>(device, input_view, output_view)
            }
            UnaryOp::EluGrad => operations.unary_into::<hephaestus_core::EluGradOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Tan => {
                operations.unary_into::<hephaestus_core::TanOp, N>(device, input_view, output_view)
            }
            UnaryOp::Asin => {
                operations.unary_into::<hephaestus_core::AsinOp, N>(device, input_view, output_view)
            }
            UnaryOp::Acos => {
                operations.unary_into::<hephaestus_core::AcosOp, N>(device, input_view, output_view)
            }
            UnaryOp::Atan => {
                operations.unary_into::<hephaestus_core::AtanOp, N>(device, input_view, output_view)
            }
            UnaryOp::Sinh => {
                operations.unary_into::<hephaestus_core::SinhOp, N>(device, input_view, output_view)
            }
            UnaryOp::Cosh => {
                operations.unary_into::<hephaestus_core::CoshOp, N>(device, input_view, output_view)
            }
            UnaryOp::Log2 => {
                operations.unary_into::<hephaestus_core::Log2Op, N>(device, input_view, output_view)
            }
            UnaryOp::Log10 => operations.unary_into::<hephaestus_core::Log10Op, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Exp2 => {
                operations.unary_into::<hephaestus_core::Exp2Op, N>(device, input_view, output_view)
            }
            UnaryOp::Atanh => operations.unary_into::<hephaestus_core::AtanhOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Asinh => operations.unary_into::<hephaestus_core::AsinhOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Acosh => operations.unary_into::<hephaestus_core::AcoshOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Expm1 => operations.unary_into::<hephaestus_core::Expm1Op, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Log1p => operations.unary_into::<hephaestus_core::Log1pOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Sign => {
                operations.unary_into::<hephaestus_core::SignOp, N>(device, input_view, output_view)
            }
            UnaryOp::Floor => operations.unary_into::<hephaestus_core::FloorOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Ceil => {
                operations.unary_into::<hephaestus_core::CeilOp, N>(device, input_view, output_view)
            }
            UnaryOp::Round => operations.unary_into::<hephaestus_core::RoundOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Trunc => operations.unary_into::<hephaestus_core::TruncOp, N>(
                device,
                input_view,
                output_view,
            ),
            UnaryOp::Erf => {
                operations.unary_into::<hephaestus_core::ErfOp, N>(device, input_view, output_view)
            }
            UnaryOp::Erfc => {
                operations.unary_into::<hephaestus_core::ErfcOp, N>(device, input_view, output_view)
            }
            UnaryOp::Lgamma => operations.unary_into::<hephaestus_core::LgammaOp, N>(
                device,
                input_view,
                output_view,
            ),
            _ => Err(unsupported_unary_operation(operation)),
        }
    }
}
