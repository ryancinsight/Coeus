//! Provider-facing elementwise operation contracts.

use super::dispatch::{BinaryElementwiseDispatch, ScalarPowerDispatch, UnaryElementwiseDispatch};
use crate::reduction::{HephaestusProvider, RankedOperand};
use coeus_core::{Float, Scalar};
use coeus_ops::{BinaryOp, UnaryOp};
use hephaestus_core::{
    CeluGradOp, CeluOp, ComputeDevice, ElementwiseOps as HephaestusElementwiseOps,
    HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp, LeakyReluGradOp, LeakyReluOp,
    ParameterizedUnaryExpr, ParameterizedUnaryOps, SoftshrinkGradOp, SoftshrinkOp, StridedView,
    ThresholdGradOp, ThresholdOp,
};

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
        UnaryOp::LeakyRelu(_) => operations.parameterized_unary_into::<LeakyReluOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::LeakyReluGrad(_) => operations.parameterized_unary_into::<LeakyReluGradOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::Hardshrink(_) => operations.parameterized_unary_into::<HardshrinkOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::HardshrinkGrad(_) => operations.parameterized_unary_into::<HardshrinkGradOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::Softshrink(_) => operations.parameterized_unary_into::<SoftshrinkOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::SoftshrinkGrad(_) => operations.parameterized_unary_into::<SoftshrinkGradOp, N>(
            P::device(),
            input,
            parameters,
            output,
        ),
        UnaryOp::Celu(_) => {
            operations.parameterized_unary_into::<CeluOp, N>(P::device(), input, parameters, output)
        }
        UnaryOp::CeluGrad(_) => operations.parameterized_unary_into::<CeluGradOp, N>(
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
