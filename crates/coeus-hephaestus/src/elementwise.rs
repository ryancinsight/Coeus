//! Generic ranked elementwise dispatch over Hephaestus providers.

use crate::{
    error::HephaestusBackendError,
    layout::ranked,
    reduction::{HephaestusBackend, HephaestusProvider, RankedOperand},
    storage::HephaestusStorage,
};
use coeus_core::{BackendError, Layout, Scalar};
use coeus_ops::{BinaryOp, ElementwiseOps, UnaryOp};
use hephaestus_core::ComputeDevice;

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
