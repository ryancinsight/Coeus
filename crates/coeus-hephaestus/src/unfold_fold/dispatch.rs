use super::provider::UnfoldFoldBackend;
use crate::layout::ranked_exact;
use coeus_core::{Layout, Scalar};
use hephaestus_core::{
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands, StridedView,
};
use leto::WindowParameters;

pub(super) fn parameters<B, T, const S: usize>(
    operation: &'static str,
    kernel: [usize; S],
    stride: [usize; S],
    padding: [usize; S],
    dilation: [usize; S],
) -> Result<WindowParameters<S>, B::Error>
where
    B: UnfoldFoldBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    WindowParameters::new(kernel, stride, padding, dilation)
        .map_err(|error| B::unfold_fold_configuration_error(operation, error.to_string()))
}

pub(super) fn unfold<B, T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&B::DeviceBuffer<T>, &Layout),
    parameters: WindowParameters<S>,
    output: (&B::DeviceBuffer<T>, &Layout),
) -> Result<(), B::Error>
where
    B: UnfoldFoldBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let input_layout = ranked_exact::<R>(operation, input.1)?;
    let output_layout = ranked_exact::<3>(operation, output.1)?;
    let operands = SlidingWindowUnfoldOperands {
        input: StridedView::new(B::unfold_fold_buffer(input.0), &input_layout),
        output: StridedView::new(B::unfold_fold_buffer(output.0), &output_layout),
    };
    B::Operations::default()
        .unfold_into(B::unfold_fold_device(), operands, parameters)
        .map_err(|source| B::unfold_fold_dispatch_error(operation, source))
}

pub(super) fn fold<B, T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&B::DeviceBuffer<T>, &Layout),
    output_spatial_shape: [usize; S],
    parameters: WindowParameters<S>,
    output: (&B::DeviceBuffer<T>, &Layout),
) -> Result<(), B::Error>
where
    B: UnfoldFoldBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let input_layout = ranked_exact::<3>(operation, input.1)?;
    let output_layout = ranked_exact::<R>(operation, output.1)?;
    let operands = SlidingWindowFoldOperands {
        input: StridedView::new(B::unfold_fold_buffer(input.0), &input_layout),
        output: StridedView::new(B::unfold_fold_buffer(output.0), &output_layout),
    };
    B::Operations::default()
        .fold_into(
            B::unfold_fold_device(),
            operands,
            output_spatial_shape,
            parameters,
        )
        .map_err(|source| B::unfold_fold_dispatch_error(operation, source))
}
