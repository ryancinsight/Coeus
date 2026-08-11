//! Generic ranked elementwise dispatch over Hephaestus providers.

use crate::{
    error::HephaestusBackendError,
    layout::ranked,
    reduction::{HephaestusBackend, HephaestusProvider, RankedOperand},
    storage::HephaestusStorage,
};
use coeus_core::{BackendError, Float, Layout, Scalar};
use coeus_ops::{BinaryOp, ElementwiseOps, ScalarPowerOps, UnaryOp};
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
pub trait BinaryElementwiseDispatch<D: ComputeDevice, T: bytemuck::Pod> {
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
    T: bytemuck::Pod + DialectScalar<E::Dialect>,
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
pub trait ScalarPowerDispatch<D: ComputeDevice, T: bytemuck::Pod> {
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
    T: bytemuck::Pod + DialectScalar<E::Dialect>,
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
pub trait UnaryElementwiseDispatch<P, T: bytemuck::Pod, E>
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
    T: bytemuck::Pod + DialectScalar<E::Dialect>,
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

/// Provider implementation of the common ranked elementwise operation set.
pub trait ElementwiseProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Provider-owned generic elementwise kernels.
    type Operations: HephaestusElementwiseOps<Self::Device, T>
        + BinaryElementwiseDispatch<Self::Device, T>
        + Default;

    /// Scalar-specific unary operation selection.
    type UnaryOperations: UnaryElementwiseDispatch<Self, T, Self::Operations>;

    /// Execute a binary operation over a fixed-rank strided output.
    fn binary<const N: usize>(
        device: &Self::Device,
        operation: BinaryOp,
        lhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        rhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        <Self::Operations as BinaryElementwiseDispatch<Self::Device, T>>::binary(
            device, operation, lhs, rhs, output,
        )
    }

    /// Execute a unary operation over a fixed-rank strided output.
    fn unary<const N: usize>(
        device: &Self::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        <Self::UnaryOperations as UnaryElementwiseDispatch<Self, T, Self::Operations>>::unary(
            device, operation, input, output,
        )
    }
}

/// Provider implementation of scalar exponentiation over strided views.
pub trait ScalarPowerProvider<T>: HephaestusProvider
where
    T: Float + leto_ops::Scalar,
{
    /// Provider-owned generic elementwise kernels.
    type Operations: ScalarPowerDispatch<Self::Device, T> + Default;

    /// Execute `output = input.powf(exponent)` at a fixed logical rank.
    fn scalar_power<const N: usize>(
        device: &Self::Device,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        exponent: T,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()> {
        <Self::Operations as ScalarPowerDispatch<Self::Device, T>>::scalar_power(
            device, input, exponent, output,
        )
    }
}

/// Provider implementation of runtime-parameter unary activation kernels.
pub trait ParameterizedElementwiseProvider: HephaestusProvider {
    /// Device-specific implementation of the device-neutral parameterized seam.
    type Operations: ParameterizedUnaryOps<Self::Device> + Default;
}

/// Dispatch a parameterized activation through the provider-owned kernel.
///
/// # Errors
///
/// Returns the provider dispatch error, or an error when `operation` is not a
/// parameterized activation.
pub fn parameterized_unary<P, const N: usize>(
    operation: UnaryOp,
    input: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
    output: RankedOperand<'_, <P::Device as ComputeDevice>::Buffer<f32>, N>,
) -> hephaestus_core::Result<()>
where
    P: ParameterizedElementwiseProvider,
    HardtanhOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    HardtanhGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
    ThresholdGradOp:
        ParameterizedUnaryExpr<<P::Operations as ParameterizedUnaryOps<P::Device>>::Dialect>,
{
    let Some(parameters) = operation.parameter_pair() else {
        return Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("operation {operation:?} is not a parameterized activation"),
        });
    };
    let operations = P::Operations::default();
    let input = StridedView::new(input.buffer, input.layout);
    let output = StridedView::new(output.buffer, output.layout);
    match operation {
        UnaryOp::Hardtanh(_) => operations.parameterized_unary_into::<HardtanhOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::HardtanhGrad(_) => operations.parameterized_unary_into::<HardtanhGradOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::Threshold(_) => operations.parameterized_unary_into::<ThresholdOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::ThresholdGrad(_) => operations.parameterized_unary_into::<ThresholdGradOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        _ => Err(hephaestus_core::HephaestusError::DispatchFailed {
            message: format!("operation {operation:?} is not a parameterized activation"),
        }),
    }
}

fn reject_broadcast_output(operation: &'static str, layout: &Layout) -> Result<(), BackendError> {
    if layout
        .shape()
        .iter()
        .zip(layout.strides())
        .any(|(&extent, &stride)| extent > 1 && stride == 0)
    {
        return Err(BackendError::Storage {
            operation,
            reason: "output layout cannot broadcast a dimension larger than one".to_owned(),
        });
    }
    Ok(())
}

impl<P> HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    #[expect(
        clippy::too_many_arguments,
        reason = "dispatch preserves the common elementwise backend contract"
    )]
    fn dispatch_binary<T>(
        &self,
        operation: BinaryOp,
        lhs: &HephaestusStorage<P, T>,
        lhs_layout: &Layout,
        rhs: &HephaestusStorage<P, T>,
        rhs_layout: &Layout,
        output: &mut HephaestusStorage<P, T>,
        output_layout: &Layout,
    ) -> Result<(), HephaestusBackendError>
    where
        P: ElementwiseProvider<T>,
        T: Scalar + leto_ops::Scalar,
    {
        reject_broadcast_output("elementwise_binary", output_layout)?;
        let rank = lhs_layout
            .ndim()
            .max(rhs_layout.ndim())
            .max(output_layout.ndim());
        match rank {
            1 => self.dispatch_binary_rank::<T, 1>(
                operation,
                lhs,
                lhs_layout,
                rhs,
                rhs_layout,
                output,
                output_layout,
            ),
            2 => self.dispatch_binary_rank::<T, 2>(
                operation,
                lhs,
                lhs_layout,
                rhs,
                rhs_layout,
                output,
                output_layout,
            ),
            3 => self.dispatch_binary_rank::<T, 3>(
                operation,
                lhs,
                lhs_layout,
                rhs,
                rhs_layout,
                output,
                output_layout,
            ),
            4 => self.dispatch_binary_rank::<T, 4>(
                operation,
                lhs,
                lhs_layout,
                rhs,
                rhs_layout,
                output,
                output_layout,
            ),
            rank => Err(BackendError::UnsupportedRank {
                operation: "elementwise_binary",
                rank,
                max_rank: 4,
            }
            .into()),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatch preserves the common elementwise backend contract"
    )]
    fn dispatch_binary_rank<T, const N: usize>(
        &self,
        operation: BinaryOp,
        lhs: &HephaestusStorage<P, T>,
        lhs_layout: &Layout,
        rhs: &HephaestusStorage<P, T>,
        rhs_layout: &Layout,
        output: &mut HephaestusStorage<P, T>,
        output_layout: &Layout,
    ) -> Result<(), HephaestusBackendError>
    where
        P: ElementwiseProvider<T>,
        T: Scalar + leto_ops::Scalar,
    {
        let lhs_layout = ranked::<N>("elementwise_binary", lhs_layout)?;
        let rhs_layout = ranked::<N>("elementwise_binary", rhs_layout)?;
        let output_layout = ranked::<N>("elementwise_binary", output_layout)?;
        P::binary(
            P::device(),
            operation,
            RankedOperand {
                buffer: lhs.buffer(),
                layout: &lhs_layout,
            },
            RankedOperand {
                buffer: rhs.buffer(),
                layout: &rhs_layout,
            },
            RankedOperand {
                buffer: output.buffer(),
                layout: &output_layout,
            },
        )
        .map_err(|source| HephaestusBackendError::device("elementwise_binary", source))
    }

    fn dispatch_unary<T>(
        &self,
        operation: UnaryOp,
        input: &HephaestusStorage<P, T>,
        input_layout: &Layout,
        output: &mut HephaestusStorage<P, T>,
        output_layout: &Layout,
    ) -> Result<(), HephaestusBackendError>
    where
        P: ElementwiseProvider<T>,
        T: Scalar + leto_ops::Scalar,
    {
        reject_broadcast_output("elementwise_unary", output_layout)?;
        let rank = input_layout.ndim().max(output_layout.ndim());
        match rank {
            1 => self.dispatch_unary_rank::<T, 1>(
                operation,
                input,
                input_layout,
                output,
                output_layout,
            ),
            2 => self.dispatch_unary_rank::<T, 2>(
                operation,
                input,
                input_layout,
                output,
                output_layout,
            ),
            3 => self.dispatch_unary_rank::<T, 3>(
                operation,
                input,
                input_layout,
                output,
                output_layout,
            ),
            4 => self.dispatch_unary_rank::<T, 4>(
                operation,
                input,
                input_layout,
                output,
                output_layout,
            ),
            rank => Err(BackendError::UnsupportedRank {
                operation: "elementwise_unary",
                rank,
                max_rank: 4,
            }
            .into()),
        }
    }

    fn dispatch_unary_rank<T, const N: usize>(
        &self,
        operation: UnaryOp,
        input: &HephaestusStorage<P, T>,
        input_layout: &Layout,
        output: &mut HephaestusStorage<P, T>,
        output_layout: &Layout,
    ) -> Result<(), HephaestusBackendError>
    where
        P: ElementwiseProvider<T>,
        T: Scalar + leto_ops::Scalar,
    {
        let input_layout = ranked::<N>("elementwise_unary", input_layout)?;
        let output_layout = ranked::<N>("elementwise_unary", output_layout)?;
        P::unary(
            P::device(),
            operation,
            RankedOperand {
                buffer: input.buffer(),
                layout: &input_layout,
            },
            RankedOperand {
                buffer: output.buffer(),
                layout: &output_layout,
            },
        )
        .map_err(|source| HephaestusBackendError::device("elementwise_unary", source))
    }
}

impl<P, T> ScalarPowerOps<T> for HephaestusBackend<P>
where
    P: ScalarPowerProvider<T>,
    T: Float + leto_ops::Scalar,
{
    fn elementwise_pow_scalar(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        exponent: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        reject_broadcast_output("elementwise scalar power", output_layout)?;
        let rank = input_layout.ndim().max(output_layout.ndim());
        match rank {
            1 => self.dispatch_scalar_power_rank::<T, 1>(
                input,
                input_layout,
                exponent,
                output,
                output_layout,
            ),
            2 => self.dispatch_scalar_power_rank::<T, 2>(
                input,
                input_layout,
                exponent,
                output,
                output_layout,
            ),
            3 => self.dispatch_scalar_power_rank::<T, 3>(
                input,
                input_layout,
                exponent,
                output,
                output_layout,
            ),
            4 => self.dispatch_scalar_power_rank::<T, 4>(
                input,
                input_layout,
                exponent,
                output,
                output_layout,
            ),
            rank => Err(BackendError::UnsupportedRank {
                operation: "elementwise scalar power",
                rank,
                max_rank: 4,
            }
            .into()),
        }
    }
}

impl<P> HephaestusBackend<P>
where
    P: HephaestusProvider,
{
    fn dispatch_scalar_power_rank<T, const N: usize>(
        &self,
        input: &HephaestusStorage<P, T>,
        input_layout: &Layout,
        exponent: T,
        output: &mut HephaestusStorage<P, T>,
        output_layout: &Layout,
    ) -> Result<(), HephaestusBackendError>
    where
        P: ScalarPowerProvider<T>,
        T: Float + leto_ops::Scalar,
    {
        let input_layout = ranked::<N>("elementwise scalar power", input_layout)?;
        let output_layout = ranked::<N>("elementwise scalar power", output_layout)?;
        P::scalar_power(
            P::device(),
            RankedOperand {
                buffer: input.buffer(),
                layout: &input_layout,
            },
            exponent,
            RankedOperand {
                buffer: output.buffer(),
                layout: &output_layout,
            },
        )
        .map_err(|source| HephaestusBackendError::device("elementwise scalar power", source))
    }
}

impl<P, T> ElementwiseOps<T> for HephaestusBackend<P>
where
    P: ElementwiseProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn elementwise_binary(
        &self,
        operation: BinaryOp,
        lhs: &Self::DeviceBuffer<T>,
        lhs_layout: &Layout,
        rhs: &Self::DeviceBuffer<T>,
        rhs_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.dispatch_binary(
            operation,
            lhs,
            lhs_layout,
            rhs,
            rhs_layout,
            output,
            output_layout,
        )
    }

    fn elementwise_unary(
        &self,
        operation: UnaryOp,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.dispatch_unary(operation, input, input_layout, output, output_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::{ranked, reject_broadcast_output};
    use coeus_core::{BackendError, Layout};

    #[test]
    fn output_broadcast_is_rejected_before_provider_dispatch() {
        let layout = Layout::from_shape_strides(vec![2, 3].into(), vec![0, 1].into(), 0);
        let error = reject_broadcast_output("elementwise_binary", &layout)
            .expect_err("broadcast output must be rejected");
        assert!(error.to_string().contains("output layout"));
    }

    #[test]
    fn rank_above_four_is_rejected_as_typed_backend_error() {
        let layout = Layout::new([1, 1, 1, 1, 1].into());
        let error = match ranked::<4>("elementwise_binary", &layout) {
            Ok(_) => panic!("rank five must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            BackendError::UnsupportedRank {
                operation: "elementwise_binary",
                rank: 5,
                max_rank: 4,
            }
        );
    }
}
