//! Provider-neutral operation dispatch.

use super::provider::{parameterized_unary, ParameterizedElementwiseProvider};
use crate::reduction::{HephaestusProvider, RankedOperand};
use coeus_ops::{BinaryOp, UnaryOp};
use hephaestus_core::{
    BinaryExpr, CeluGradOp, CeluOp, ComputeDevice, DialectScalar,
    ElementwiseOps as HephaestusElementwiseOps, HardshrinkGradOp, HardshrinkOp, HardsigmoidGradOp,
    HardsigmoidOp, HardswishGradOp, HardswishOp, HardtanhGradOp, HardtanhOp, LeakyReluGradOp,
    LeakyReluOp, ParameterizedUnaryExpr, ParameterizedUnaryOps, SoftshrinkGradOp, SoftshrinkOp,
    SoftsignGradOp, SoftsignOp, StridedView, ThresholdGradOp, ThresholdOp, TypedBinaryExpr,
    UnaryExpr,
};

fn unsupported_unary_operation(operation: UnaryOp) -> hephaestus_core::HephaestusError {
    hephaestus_core::HephaestusError::DispatchFailed {
        message: format!(
            "unary elementwise operation {operation:?} is not implemented by provider"
        ),
    }
}

macro_rules! dispatch_unary_into {
    ($operation:expr, $operations:expr, $device:expr, $input:expr, $output:expr, { $($variant:path => $kernel:ty),+ $(,)? }) => {
        match $operation {
            $(
                $variant => Some($operations.unary_into::<$kernel, N>($device, $input, $output)),
            )+
            _ => None,
        }
    };
}

#[inline(always)]
fn dispatch_core_unary_operations<D, E, T, const N: usize>(
    operations: &E,
    device: &D,
    operation: UnaryOp,
    input: StridedView<'_, D::Buffer<T>, N>,
    output: StridedView<'_, D::Buffer<T>, N>,
) -> Option<hephaestus_core::Result<()>>
where
    D: ComputeDevice,
    T: eunomia::Pod + DialectScalar<E::Dialect>,
    E: HephaestusElementwiseOps<D, T>,
    hephaestus_core::SinOp: UnaryExpr<E::Dialect>,
    hephaestus_core::CosOp: UnaryExpr<E::Dialect>,
    hephaestus_core::ExpOp: UnaryExpr<E::Dialect>,
    hephaestus_core::LnOp: UnaryExpr<E::Dialect>,
    hephaestus_core::NegOp: UnaryExpr<E::Dialect>,
    hephaestus_core::AbsOp: UnaryExpr<E::Dialect>,
    hephaestus_core::SqrtOp: UnaryExpr<E::Dialect>,
    hephaestus_core::RecipOp: UnaryExpr<E::Dialect>,
{
    match operation {
        UnaryOp::Sin => {
            Some(operations.unary_into::<hephaestus_core::SinOp, N>(device, input, output))
        }
        UnaryOp::Cos => {
            Some(operations.unary_into::<hephaestus_core::CosOp, N>(device, input, output))
        }
        UnaryOp::Exp => {
            Some(operations.unary_into::<hephaestus_core::ExpOp, N>(device, input, output))
        }
        UnaryOp::Log => {
            Some(operations.unary_into::<hephaestus_core::LnOp, N>(device, input, output))
        }
        UnaryOp::Neg => {
            Some(operations.unary_into::<hephaestus_core::NegOp, N>(device, input, output))
        }
        UnaryOp::Abs => {
            Some(operations.unary_into::<hephaestus_core::AbsOp, N>(device, input, output))
        }
        UnaryOp::Sqrt => {
            Some(operations.unary_into::<hephaestus_core::SqrtOp, N>(device, input, output))
        }
        UnaryOp::Recip => {
            Some(operations.unary_into::<hephaestus_core::RecipOp, N>(device, input, output))
        }
        _ => None,
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
        if let Some(result) = dispatch_core_unary_operations::<P::Device, E, T, N>(
            &operations,
            device,
            operation,
            input,
            output,
        ) {
            result
        } else {
            Err(unsupported_unary_operation(operation))
        }
    }
}

impl<P, T, E> UnaryElementwiseDispatch<P, T, E> for ActivationUnaryOperations
where
    P: HephaestusProvider,
    E: HephaestusElementwiseOps<P::Device, T> + Default,
    T: eunomia::Pod + DialectScalar<E::Dialect>,
    ActivationParameter<T>: ActivationParameterDispatch<P, T>,
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
    HardsigmoidOp: UnaryExpr<E::Dialect>,
    HardsigmoidGradOp: UnaryExpr<E::Dialect>,
    HardswishOp: UnaryExpr<E::Dialect>,
    HardswishGradOp: UnaryExpr<E::Dialect>,
    SoftsignOp: UnaryExpr<E::Dialect>,
    SoftsignGradOp: UnaryExpr<E::Dialect>,
{
    fn unary<const N: usize>(
        device: &P::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        let input_view = StridedView::new(input.buffer, input.layout);
        let output_view = StridedView::new(output.buffer, output.layout);
        let operations = E::default();
        if let Some(result) = dispatch_core_unary_operations::<P::Device, E, T, N>(
            &operations,
            device,
            operation,
            input_view,
            output_view,
        ) {
            return result;
        }
        if let Some(result) = dispatch_unary_into!(
            operation,
            operations,
            device,
            input_view,
            output_view,
            {
                UnaryOp::Relu => hephaestus_core::ReluOp,
                UnaryOp::ReluGrad => hephaestus_core::ReluGradOp,
                UnaryOp::Sigmoid => hephaestus_core::SigmoidOp,
                UnaryOp::SigmoidGrad => hephaestus_core::SigmoidGradOp,
                UnaryOp::Tanh => hephaestus_core::TanhOp,
                UnaryOp::TanhGrad => hephaestus_core::TanhGradOp,
                UnaryOp::Gelu => hephaestus_core::GeluOp,
                UnaryOp::GeluGrad => hephaestus_core::GeluGradOp,
                UnaryOp::GeluTanh => hephaestus_core::GeluTanhOp,
                UnaryOp::GeluTanhGrad => hephaestus_core::GeluTanhGradOp,
                UnaryOp::Silu => hephaestus_core::SiluOp,
                UnaryOp::SiluGrad => hephaestus_core::SiluGradOp,
                UnaryOp::Softplus => hephaestus_core::SoftplusOp,
                UnaryOp::SoftplusGrad => hephaestus_core::SoftplusGradOp,
                UnaryOp::Mish => hephaestus_core::MishOp,
                UnaryOp::MishGrad => hephaestus_core::MishGradOp,
                UnaryOp::Elu => hephaestus_core::EluOp,
                UnaryOp::EluGrad => hephaestus_core::EluGradOp,
                UnaryOp::Hardsigmoid => HardsigmoidOp,
                UnaryOp::HardsigmoidGrad => HardsigmoidGradOp,
                UnaryOp::Hardswish => HardswishOp,
                UnaryOp::HardswishGrad => HardswishGradOp,
                UnaryOp::Softsign => SoftsignOp,
                UnaryOp::SoftsignGrad => SoftsignGradOp,
                UnaryOp::Tan => hephaestus_core::TanOp,
                UnaryOp::Asin => hephaestus_core::AsinOp,
                UnaryOp::Acos => hephaestus_core::AcosOp,
                UnaryOp::Atan => hephaestus_core::AtanOp,
                UnaryOp::Sinh => hephaestus_core::SinhOp,
                UnaryOp::Cosh => hephaestus_core::CoshOp,
                UnaryOp::Log2 => hephaestus_core::Log2Op,
                UnaryOp::Log10 => hephaestus_core::Log10Op,
                UnaryOp::Exp2 => hephaestus_core::Exp2Op,
                UnaryOp::Atanh => hephaestus_core::AtanhOp,
                UnaryOp::Asinh => hephaestus_core::AsinhOp,
                UnaryOp::Acosh => hephaestus_core::AcoshOp,
                UnaryOp::Expm1 => hephaestus_core::Expm1Op,
                UnaryOp::Log1p => hephaestus_core::Log1pOp,
                UnaryOp::Sign => hephaestus_core::SignOp,
                UnaryOp::Floor => hephaestus_core::FloorOp,
                UnaryOp::Ceil => hephaestus_core::CeilOp,
                UnaryOp::Round => hephaestus_core::RoundOp,
                UnaryOp::Trunc => hephaestus_core::TruncOp,
                UnaryOp::Erf => hephaestus_core::ErfOp,
                UnaryOp::Erfc => hephaestus_core::ErfcOp,
                UnaryOp::Lgamma => hephaestus_core::LgammaOp
            }
        ) {
            return result;
        }
        match operation {
            UnaryOp::Hardtanh(_)
            | UnaryOp::HardtanhGrad(_)
            | UnaryOp::LeakyRelu(_)
            | UnaryOp::LeakyReluGrad(_)
            | UnaryOp::Hardshrink(_)
            | UnaryOp::HardshrinkGrad(_)
            | UnaryOp::Softshrink(_)
            | UnaryOp::SoftshrinkGrad(_)
            | UnaryOp::Threshold(_)
            | UnaryOp::ThresholdGrad(_)
            | UnaryOp::Celu(_)
            | UnaryOp::CeluGrad(_) => <ActivationParameter<T> as ActivationParameterDispatch<
                P,
                T,
            >>::dispatch(operation, input, output),
            UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Neg
            | UnaryOp::Abs
            | UnaryOp::Sqrt
            | UnaryOp::Recip => unreachable!("handled by core unary dispatch"),
            _ => Err(unsupported_unary_operation(operation)),
        }
    }
}

/// Routes parameterized activations (hardtanh, leaky_relu, ...) for the
/// activation unary set.
///
/// The provider parameterized kernel surface is f32-only
/// (`ParameterizedUnaryOps::parameterized_unary_into` takes `Buffer<f32>`
/// with `[f32; 2]` parameters), so only `f32` dispatches on-device; every
/// other scalar reports the operation unsupported until the core trait
/// generalizes over the buffer element type.
trait ActivationParameterDispatch<P: HephaestusProvider, T: eunomia::Pod> {
    fn dispatch<const N: usize>(
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
}

struct ActivationParameter<T>(core::marker::PhantomData<T>);

impl<P> ActivationParameterDispatch<P, f32> for ActivationParameter<f32>
where
    P: ParameterizedElementwiseProvider,
    HardtanhOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    HardtanhGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    LeakyReluOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    LeakyReluGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    HardshrinkOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    HardshrinkGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    SoftshrinkOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    SoftshrinkGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    CeluOp: ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    CeluGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
{
    fn dispatch<const N: usize>(
        operation: UnaryOp,
        input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
        output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
    ) -> hephaestus_core::Result<()> {
        parameterized_unary::<P, N>(operation, input, output)
    }
}

impl<P> ActivationParameterDispatch<P, f64> for ActivationParameter<f64>
where
    P: HephaestusProvider,
{
    fn dispatch<const N: usize>(
        operation: UnaryOp,
        _input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f64>, N>,
        _output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f64>, N>,
    ) -> hephaestus_core::Result<()> {
        Err(unsupported_unary_operation(operation))
    }
}
