use super::provider::PoolingBackend;
use crate::layout::ranked_exact;
use coeus_core::{Layout, Scalar};
use hephaestus_core::{
    PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps, StridedView,
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
    B: PoolingBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    WindowParameters::new(kernel, stride, padding, dilation)
        .map_err(|error| B::pooling_configuration_error(operation, error.to_string()))
}

pub(super) fn forward<B, T, const R: usize, const S: usize>(
    operation: &'static str,
    input: (&B::DeviceBuffer<T>, &Layout),
    parameters: WindowParameters<S>,
    mode: PoolingMode,
    output: (&B::DeviceBuffer<T>, &Layout),
) -> Result<(), B::Error>
where
    B: PoolingBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let input_layout = ranked_exact::<R>(operation, input.1)?;
    let output_layout = ranked_exact::<R>(operation, output.1)?;
    let operands = PoolingForwardOperands {
        input: StridedView::new(B::pooling_buffer(input.0), &input_layout),
        output: StridedView::new(B::pooling_buffer(output.0), &output_layout),
    };
    B::Operations::default()
        .pooling_forward_into(B::pooling_device(), operands, parameters, mode)
        .map_err(|source| B::pooling_dispatch_error(operation, source))
}

pub(super) fn backward<B, T, const R: usize, const S: usize>(
    operation: &'static str,
    grad_output: (&B::DeviceBuffer<T>, &Layout),
    input: Option<(&B::DeviceBuffer<T>, &Layout)>,
    parameters: WindowParameters<S>,
    mode: PoolingMode,
    grad_input: (&B::DeviceBuffer<T>, &Layout),
) -> Result<(), B::Error>
where
    B: PoolingBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let grad_output_layout = ranked_exact::<R>(operation, grad_output.1)?;
    let grad_input_layout = ranked_exact::<R>(operation, grad_input.1)?;
    let input_layout = input
        .map(|(_, layout)| ranked_exact::<R>(operation, layout))
        .transpose()?;
    let input = input
        .zip(input_layout.as_ref())
        .map(|((buffer, _), layout)| StridedView::new(B::pooling_buffer(buffer), layout));
    let operands = PoolingBackwardOperands {
        input,
        grad_output: StridedView::new(B::pooling_buffer(grad_output.0), &grad_output_layout),
        grad_input: StridedView::new(B::pooling_buffer(grad_input.0), &grad_input_layout),
    };
    B::Operations::default()
        .pooling_backward_accumulate(B::pooling_device(), operands, parameters, mode)
        .map_err(|source| B::pooling_dispatch_error(operation, source))
}
