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
    ComputeDevice, HardtanhGradOp, HardtanhOp, ParameterizedUnaryExpr, ParameterizedUnaryOps,
    StridedView, ThresholdGradOp, ThresholdOp,
};

/// Provider implementation of the common ranked elementwise operation set.
pub trait ElementwiseProvider<T>: HephaestusProvider
where
    T: Scalar + leto_ops::Scalar,
{
    /// Execute a binary operation over a fixed-rank strided output.
    fn binary<const N: usize>(
        device: &Self::Device,
        operation: BinaryOp,
        lhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        rhs: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;

    /// Execute a unary operation over a fixed-rank strided output.
    fn unary<const N: usize>(
        device: &Self::Device,
        operation: UnaryOp,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
}

/// Provider implementation of scalar exponentiation over strided views.
pub trait ScalarPowerProvider<T>: HephaestusProvider
where
    T: Float + leto_ops::Scalar,
{
    /// Execute `output = input.powf(exponent)` at a fixed logical rank.
    fn scalar_power<const N: usize>(
        device: &Self::Device,
        input: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
        exponent: T,
        output: RankedOperand<'_, <Self::Device as ComputeDevice>::Buffer<T>, N>,
    ) -> hephaestus_core::Result<()>;
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
