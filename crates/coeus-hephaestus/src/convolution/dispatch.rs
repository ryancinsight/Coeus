use crate::{convolution::provider::ConvolutionBackend, layout::ranked};
use coeus_core::{Layout, Scalar};
use coeus_ops::{
    ConvolutionBackward as CoeusConvolutionBackward, ConvolutionForward as CoeusConvolutionForward,
};
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews,
    ConvolutionOps, DeviceBuffer, StridedView,
};
use leto::{ConvolutionParameters, Layout as LetoLayout, TransposedConvolutionParameters};

/// Borrowed Coeus operands for provider-owned forward dispatch.
pub struct Forward<'a, B, T>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    /// Input storage.
    pub input: &'a B::DeviceBuffer<T>,
    /// Input layout.
    pub input_layout: &'a Layout,
    /// Weight storage.
    pub weight: &'a B::DeviceBuffer<T>,
    /// Weight layout.
    pub weight_layout: &'a Layout,
    /// Optional bias storage.
    pub bias: Option<&'a B::DeviceBuffer<T>>,
    /// Output storage.
    pub output: &'a mut B::DeviceBuffer<T>,
    /// Output layout.
    pub output_layout: &'a Layout,
}

impl<'a, B, T> From<CoeusConvolutionForward<'a, B, T>> for Forward<'a, B, T>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn from(request: CoeusConvolutionForward<'a, B, T>) -> Self {
        Self {
            input: request.input,
            input_layout: request.input_layout,
            weight: request.weight,
            weight_layout: request.weight_layout,
            bias: request.bias,
            output: request.output,
            output_layout: request.output_layout,
        }
    }
}

/// Borrowed Coeus operands for provider-owned backward dispatch.
pub struct Backward<'a, B, T>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    /// Output-gradient storage.
    pub grad_output: &'a B::DeviceBuffer<T>,
    /// Output-gradient layout.
    pub grad_output_layout: &'a Layout,
    /// Forward input storage.
    pub input: &'a B::DeviceBuffer<T>,
    /// Forward input layout.
    pub input_layout: &'a Layout,
    /// Forward weight storage.
    pub weight: &'a B::DeviceBuffer<T>,
    /// Forward weight layout.
    pub weight_layout: &'a Layout,
    /// Optional input-gradient target.
    pub grad_input: Option<&'a mut B::DeviceBuffer<T>>,
    /// Input-gradient layout.
    pub grad_input_layout: &'a Layout,
    /// Optional weight-gradient target.
    pub grad_weight: Option<&'a mut B::DeviceBuffer<T>>,
    /// Weight-gradient layout.
    pub grad_weight_layout: &'a Layout,
    /// Optional bias-gradient target.
    pub grad_bias: Option<&'a mut B::DeviceBuffer<T>>,
}

impl<'a, B, T> From<CoeusConvolutionBackward<'a, B, T>> for Backward<'a, B, T>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    fn from(request: CoeusConvolutionBackward<'a, B, T>) -> Self {
        Self {
            grad_output: request.grad_output,
            grad_output_layout: request.grad_output_layout,
            input: request.input,
            input_layout: request.input_layout,
            weight: request.weight,
            weight_layout: request.weight_layout,
            grad_input: request.grad_input,
            grad_input_layout: request.grad_input_layout,
            grad_weight: request.grad_weight,
            grad_weight_layout: request.grad_weight_layout,
            grad_bias: request.grad_bias,
        }
    }
}

fn configuration<B, T>(operation: &'static str, error: impl std::fmt::Display) -> B::Error
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    B::convolution_configuration_error(operation, error.to_string())
}

fn bias_layout<B, T>(
    operation: &'static str,
    storage: &B::DeviceBuffer<T>,
) -> Result<LetoLayout<1>, B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    LetoLayout::c_contiguous([B::convolution_buffer(storage).len()])
        .map_err(|error| configuration::<B, T>(operation, error))
}

/// Dispatch regular convolution through the selected provider.
///
/// # Errors
///
/// Returns the backend error for invalid layouts, parameters, or provider
/// execution failures.
pub fn regular_forward<B, T, const R: usize, const D: usize>(
    request: Forward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    forward::<B, T, R, D>(
        request,
        ConvolutionKind::Regular(ConvolutionParameters::new(stride, padding, dilation)),
    )
}

/// Dispatch regular-convolution backward through the selected provider.
///
/// # Errors
///
/// Returns the backend error for invalid layouts, parameters, or provider
/// execution failures.
pub fn regular_backward<B, T, const R: usize, const D: usize>(
    request: Backward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    backward::<B, T, R, D>(
        request,
        ConvolutionKind::Regular(ConvolutionParameters::new(stride, padding, dilation)),
    )
}

/// Dispatch transposed convolution through the selected provider.
///
/// # Errors
///
/// Returns the backend error for invalid layouts, parameters, or provider
/// execution failures.
pub fn transposed_forward<B, T, const R: usize, const D: usize>(
    request: Forward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    forward::<B, T, R, D>(
        request,
        ConvolutionKind::Transposed(TransposedConvolutionParameters::new(
            stride,
            padding,
            output_padding,
            dilation,
        )),
    )
}

/// Dispatch transposed-convolution backward through the selected provider.
///
/// # Errors
///
/// Returns the backend error for invalid layouts, parameters, or provider
/// execution failures.
pub fn transposed_backward<B, T, const R: usize, const D: usize>(
    request: Backward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    backward::<B, T, R, D>(
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

fn forward<B, T, const R: usize, const D: usize>(
    request: Forward<'_, B, T>,
    kind: ConvolutionKind<D>,
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
    T: Scalar + leto_ops::Scalar,
{
    let operation = match kind {
        ConvolutionKind::Regular(_) => "convolution",
        ConvolutionKind::Transposed(_) => "transposed convolution",
    };
    let input_layout = ranked::<R>(operation, request.input_layout)?;
    let weight_layout = ranked::<R>(operation, request.weight_layout)?;
    let output_layout = ranked::<R>(operation, request.output_layout)?;
    let bias_layout = request
        .bias
        .map(|bias| bias_layout::<B, T>(operation, bias))
        .transpose()?;
    let operands = ConvolutionForwardOperands {
        input: StridedView::new(B::convolution_buffer(request.input), &input_layout),
        weight: StridedView::new(B::convolution_buffer(request.weight), &weight_layout),
        bias: request
            .bias
            .zip(bias_layout.as_ref())
            .map(|(bias, layout)| StridedView::new(B::convolution_buffer(bias), layout)),
        output: StridedView::new(B::convolution_buffer(request.output), &output_layout),
    };
    let operations = B::Operations::default();
    let result = match kind {
        ConvolutionKind::Regular(parameters) => operations.convolution_forward_into(
            B::convolution_device(),
            operands,
            parameters.map_err(|error| configuration::<B, T>(operation, error))?,
        ),
        ConvolutionKind::Transposed(parameters) => operations.convolution_transposed_forward_into(
            B::convolution_device(),
            operands,
            parameters.map_err(|error| configuration::<B, T>(operation, error))?,
        ),
    };
    result.map_err(|source| B::convolution_dispatch_error(operation, source))
}

fn backward<B, T, const R: usize, const D: usize>(
    request: Backward<'_, B, T>,
    kind: ConvolutionKind<D>,
) -> Result<(), B::Error>
where
    B: ConvolutionBackend<T>,
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
        .map(|bias| bias_layout::<B, T>(operation, bias))
        .transpose()?;
    let operands = ConvolutionBackwardOperands {
        input: StridedView::new(B::convolution_buffer(request.input), &input_layout),
        weight: StridedView::new(B::convolution_buffer(request.weight), &weight_layout),
        grad_output: StridedView::new(
            B::convolution_buffer(request.grad_output),
            &grad_output_layout,
        ),
        gradients: ConvolutionGradientViews {
            input: request
                .grad_input
                .zip(grad_input_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(B::convolution_buffer(buffer), layout)),
            weight: request
                .grad_weight
                .zip(grad_weight_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(B::convolution_buffer(buffer), layout)),
            bias: request
                .grad_bias
                .zip(grad_bias_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(B::convolution_buffer(buffer), layout)),
        },
    };
    let operations = B::Operations::default();
    let result = match kind {
        ConvolutionKind::Regular(parameters) => operations.convolution_backward_accumulate(
            B::convolution_device(),
            operands,
            parameters.map_err(|error| configuration::<B, T>(operation, error))?,
        ),
        ConvolutionKind::Transposed(parameters) => operations
            .convolution_transposed_backward_accumulate(
                B::convolution_device(),
                operands,
                parameters.map_err(|error| configuration::<B, T>(operation, error))?,
            ),
    };
    result.map_err(|source| B::convolution_dispatch_error(operation, source))
}
