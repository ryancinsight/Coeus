use crate::{
    HephaestusBackendError, HephaestusStorage, convolution::provider::ConvolutionProvider,
    layout::ranked,
};
use coeus_core::{Layout, Scalar};
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews,
    ConvolutionOps, DeviceBuffer, StridedView,
};
use leto::{ConvolutionParameters, Layout as LetoLayout, TransposedConvolutionParameters};

pub(super) struct Forward<'a, P, T>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    pub input: &'a HephaestusStorage<P, T>,
    pub input_layout: &'a Layout,
    pub weight: &'a HephaestusStorage<P, T>,
    pub weight_layout: &'a Layout,
    pub bias: Option<&'a HephaestusStorage<P, T>>,
    pub output: &'a mut HephaestusStorage<P, T>,
    pub output_layout: &'a Layout,
}

pub(super) struct Backward<'a, P, T>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    pub grad_output: &'a HephaestusStorage<P, T>,
    pub grad_output_layout: &'a Layout,
    pub input: &'a HephaestusStorage<P, T>,
    pub input_layout: &'a Layout,
    pub weight: &'a HephaestusStorage<P, T>,
    pub weight_layout: &'a Layout,
    pub grad_input: Option<&'a mut HephaestusStorage<P, T>>,
    pub grad_input_layout: &'a Layout,
    pub grad_weight: Option<&'a mut HephaestusStorage<P, T>>,
    pub grad_weight_layout: &'a Layout,
    pub grad_bias: Option<&'a mut HephaestusStorage<P, T>>,
}

fn configuration(operation: &'static str, error: impl std::fmt::Display) -> HephaestusBackendError {
    HephaestusBackendError::device(
        operation,
        hephaestus_core::HephaestusError::InvalidConfiguration {
            message: error.to_string(),
        },
    )
}

fn bias_layout<P, T>(
    operation: &'static str,
    storage: &HephaestusStorage<P, T>,
) -> Result<LetoLayout<1>, HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    LetoLayout::c_contiguous([storage.buffer().len()])
        .map_err(|error| configuration(operation, error))
}

pub(super) fn regular_forward<P, T, const R: usize, const D: usize>(
    request: Forward<'_, P, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    let input_layout = ranked::<R>("convolution", request.input_layout)?;
    let weight_layout = ranked::<R>("convolution", request.weight_layout)?;
    let output_layout = ranked::<R>("convolution", request.output_layout)?;
    let bias_layout = request
        .bias
        .map(|bias| bias_layout("convolution", bias))
        .transpose()?;
    let parameters = ConvolutionParameters::new(stride, padding, dilation)
        .map_err(|error| configuration("convolution", error))?;
    P::Operations::default()
        .convolution_forward_into(
            P::device(),
            ConvolutionForwardOperands {
                input: StridedView::new(request.input.buffer(), &input_layout),
                weight: StridedView::new(request.weight.buffer(), &weight_layout),
                bias: request
                    .bias
                    .zip(bias_layout.as_ref())
                    .map(|(bias, layout)| StridedView::new(bias.buffer(), layout)),
                output: StridedView::new(request.output.buffer(), &output_layout),
            },
            parameters,
        )
        .map_err(|source| HephaestusBackendError::device("convolution", source))
}

pub(super) fn regular_backward<P, T, const R: usize, const D: usize>(
    request: Backward<'_, P, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    backward::<P, T, R, D>(
        request,
        ConvolutionKind::Regular(ConvolutionParameters::new(stride, padding, dilation)),
    )
}

pub(super) fn transposed_forward<P, T, const R: usize, const D: usize>(
    request: Forward<'_, P, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    let input_layout = ranked::<R>("transposed convolution", request.input_layout)?;
    let weight_layout = ranked::<R>("transposed convolution", request.weight_layout)?;
    let output_layout = ranked::<R>("transposed convolution", request.output_layout)?;
    let bias_layout = request
        .bias
        .map(|bias| bias_layout("transposed convolution", bias))
        .transpose()?;
    let parameters =
        TransposedConvolutionParameters::new(stride, padding, output_padding, dilation)
            .map_err(|error| configuration("transposed convolution", error))?;
    P::Operations::default()
        .convolution_transposed_forward_into(
            P::device(),
            ConvolutionForwardOperands {
                input: StridedView::new(request.input.buffer(), &input_layout),
                weight: StridedView::new(request.weight.buffer(), &weight_layout),
                bias: request
                    .bias
                    .zip(bias_layout.as_ref())
                    .map(|(bias, layout)| StridedView::new(bias.buffer(), layout)),
                output: StridedView::new(request.output.buffer(), &output_layout),
            },
            parameters,
        )
        .map_err(|source| HephaestusBackendError::device("transposed convolution", source))
}

pub(super) fn transposed_backward<P, T, const R: usize, const D: usize>(
    request: Backward<'_, P, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    backward::<P, T, R, D>(
        request,
        ConvolutionKind::Transposed(TransposedConvolutionParameters::new(
            stride,
            padding,
            output_padding,
            dilation,
        )),
    )
}

enum ConvolutionKind<const D: usize> {
    Regular(leto::Result<ConvolutionParameters<D>>),
    Transposed(leto::Result<TransposedConvolutionParameters<D>>),
}

fn backward<P, T, const R: usize, const D: usize>(
    request: Backward<'_, P, T>,
    kind: ConvolutionKind<D>,
) -> Result<(), HephaestusBackendError>
where
    P: ConvolutionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    let operation = match kind {
        ConvolutionKind::Regular(_) => "convolution backward",
        ConvolutionKind::Transposed(_) => "transposed convolution backward",
    };
    let input_layout = ranked::<R>(operation, request.input_layout)?;
    let weight_layout = ranked::<R>(operation, request.weight_layout)?;
    let grad_output_layout = ranked::<R>(operation, request.grad_output_layout)?;
    let grad_input_layout = request
        .grad_input
        .as_ref()
        .map(|_| ranked::<R>(operation, request.grad_input_layout))
        .transpose()?;
    let grad_weight_layout = request
        .grad_weight
        .as_ref()
        .map(|_| ranked::<R>(operation, request.grad_weight_layout))
        .transpose()?;
    let grad_bias_layout = request
        .grad_bias
        .as_ref()
        .map(|bias| bias_layout(operation, bias))
        .transpose()?;
    let operands = ConvolutionBackwardOperands {
        input: StridedView::new(request.input.buffer(), &input_layout),
        weight: StridedView::new(request.weight.buffer(), &weight_layout),
        grad_output: StridedView::new(request.grad_output.buffer(), &grad_output_layout),
        gradients: ConvolutionGradientViews {
            input: request
                .grad_input
                .zip(grad_input_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer(), layout)),
            weight: request
                .grad_weight
                .zip(grad_weight_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer(), layout)),
            bias: request
                .grad_bias
                .zip(grad_bias_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer(), layout)),
        },
    };
    let operations = P::Operations::default();
    let result = match kind {
        ConvolutionKind::Regular(parameters) => operations.convolution_backward_accumulate(
            P::device(),
            operands,
            parameters.map_err(|error| configuration(operation, error))?,
        ),
        ConvolutionKind::Transposed(parameters) => operations
            .convolution_transposed_backward_accumulate(
                P::device(),
                operands,
                parameters.map_err(|error| configuration(operation, error))?,
            ),
    };
    result.map_err(|source| HephaestusBackendError::device(operation, source))
}
