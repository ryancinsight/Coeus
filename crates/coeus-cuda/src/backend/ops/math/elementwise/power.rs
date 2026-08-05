//! Provider-owned CUDA scalar-power dispatch.

use super::can_route_dynamic_strided;
use crate::backend::CudaScalar;
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::{Float, Layout};
use hephaestus_core::BlockWidth;
use std::sync::Arc;

fn ranked_layout<const N: usize>(layout: &Layout) -> Result<leto::Layout<N>, CudaBackendError> {
    let shape: [usize; N] =
        layout
            .shape()
            .try_into()
            .map_err(|_| CudaBackendError::InvalidLayout {
                operation: "elementwise scalar power",
                reason: "layout rank does not match the selected provider dispatch",
            })?;
    let strides: Vec<isize> = layout
        .strides()
        .iter()
        .copied()
        .map(|stride| {
            isize::try_from(stride).map_err(|_| CudaBackendError::InvalidLayout {
                operation: "elementwise scalar power",
                reason: "layout stride exceeds the provider index range",
            })
        })
        .collect::<Result<_, _>>()?;
    let strides: [isize; N] = strides
        .try_into()
        .map_err(|_| CudaBackendError::InvalidLayout {
            operation: "elementwise scalar power",
            reason: "layout stride rank does not match the selected provider dispatch",
        })?;
    Ok(leto::Layout::new(shape, strides, layout.offset()))
}

/// Dispatch `output = input.powf(exponent)` through Hephaestus CUDA.
pub(crate) fn elementwise_pow_scalar<T>(
    input: &CudaStorage<T>,
    input_layout: &Layout,
    exponent: T,
    output: &CudaStorage<T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: Float + CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    if Arc::ptr_eq(&input.buffer, &output.buffer) {
        return Err(CudaBackendError::InvalidLayout {
            operation: "elementwise scalar power",
            reason: "output must not alias input",
        });
    }
    if !can_route_dynamic_strided(&[input_layout], output_layout) {
        return Err(CudaBackendError::UnsupportedRank {
            operation: "elementwise scalar power",
            rank: input_layout.ndim().max(output_layout.ndim()),
            max_rank: hephaestus_cuda::MAX_STRIDED_RANK,
        });
    }

    macro_rules! dispatch_rank {
        ($rank:expr) => {{
            let input_layout = ranked_layout::<$rank>(input_layout)?;
            let output_layout = ranked_layout::<$rank>(output_layout)?;
            hephaestus_cuda::scalar_elementwise_strided_into::<hephaestus_cuda::PowOp, T, $rank>(
                crate::backend::get_cuda_device(),
                hephaestus_cuda::StridedOperand {
                    buffer: input.buffer.as_ref(),
                    layout: &input_layout,
                },
                exponent,
                hephaestus_cuda::StridedOperand {
                    buffer: output.buffer.as_ref(),
                    layout: &output_layout,
                },
                BlockWidth::DEFAULT,
            )
            .map_err(|source| CudaBackendError::dispatch("elementwise scalar power", source))
        }};
    }

    match input_layout.ndim().max(output_layout.ndim()) {
        1 => dispatch_rank!(1),
        2 => dispatch_rank!(2),
        3 => dispatch_rank!(3),
        4 => dispatch_rank!(4),
        rank => Err(CudaBackendError::UnsupportedRank {
            operation: "elementwise scalar power",
            rank,
            max_rank: hephaestus_cuda::MAX_STRIDED_RANK,
        }),
    }
}
