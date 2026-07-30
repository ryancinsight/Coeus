use crate::backend::{WgpuBackend, WgpuBackendError, WgpuScalar, get_wgpu_context};
use coeus_core::{BackendError, Float};
use coeus_ops::{ConvolutionBackward, ConvolutionForward};
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews,
    ConvolutionOps, DeviceBuffer, StridedView,
};
use hephaestus_wgpu::{DialectScalar, WgpuConvolutionOps, WgpuDevice, Wgsl};
use leto::{ConvolutionParameters, Layout, TransposedConvolutionParameters};

fn layout<const R: usize>(
    operation: &'static str,
    source: &coeus_core::Layout,
) -> Result<Layout<R>, WgpuBackendError> {
    coeus_leto::to_leto_layout(source).map_err(|error| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })
}

fn bias_layout<T: WgpuScalar>(
    operation: &'static str,
    storage: &crate::WgpuStorage<T>,
) -> Result<Layout<1>, WgpuBackendError> {
    Layout::c_contiguous([storage.buffer.len()]).map_err(|error| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })
}

impl<T> coeus_ops::ConvOps<T> for WgpuBackend
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        let input_layout = layout::<R>("convolution", request.input_layout)?;
        let weight_layout = layout::<R>("convolution", request.weight_layout)?;
        let output_layout = layout::<R>("convolution", request.output_layout)?;
        let bias_layout = request
            .bias
            .map(|bias| bias_layout("convolution", bias))
            .transpose()?;
        let parameters =
            ConvolutionParameters::new(stride, padding, dilation).map_err(|error| {
                WgpuBackendError::Validation(BackendError::Storage {
                    operation: "convolution",
                    reason: error.to_string(),
                })
            })?;
        WgpuConvolutionOps
            .convolution_forward_into(
                &get_wgpu_context().hephaestus_device,
                ConvolutionForwardOperands {
                    input: StridedView::new(request.input.buffer.as_ref(), &input_layout),
                    weight: StridedView::new(request.weight.buffer.as_ref(), &weight_layout),
                    bias: request
                        .bias
                        .zip(bias_layout.as_ref())
                        .map(|(bias, layout)| StridedView::new(bias.buffer.as_ref(), layout)),
                    output: StridedView::new(request.output.buffer.as_ref(), &output_layout),
                },
                parameters,
            )
            .map_err(|source| WgpuBackendError::dispatch("convolution", source))
    }

    fn convolution_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error> {
        backward::<T, R, D>(
            request,
            Parameters::Regular(ConvolutionParameters::new(stride, padding, dilation)),
        )
    }

    fn convolution_transposed_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        let input_layout = layout::<R>("transposed convolution", request.input_layout)?;
        let weight_layout = layout::<R>("transposed convolution", request.weight_layout)?;
        let output_layout = layout::<R>("transposed convolution", request.output_layout)?;
        let bias_layout = request
            .bias
            .map(|bias| bias_layout("transposed convolution", bias))
            .transpose()?;
        let parameters =
            TransposedConvolutionParameters::new(stride, padding, output_padding, dilation)
                .map_err(|error| {
                    WgpuBackendError::Validation(BackendError::Storage {
                        operation: "transposed convolution",
                        reason: error.to_string(),
                    })
                })?;
        WgpuConvolutionOps
            .convolution_transposed_forward_into(
                &get_wgpu_context().hephaestus_device,
                ConvolutionForwardOperands {
                    input: StridedView::new(request.input.buffer.as_ref(), &input_layout),
                    weight: StridedView::new(request.weight.buffer.as_ref(), &weight_layout),
                    bias: request
                        .bias
                        .zip(bias_layout.as_ref())
                        .map(|(bias, layout)| StridedView::new(bias.buffer.as_ref(), layout)),
                    output: StridedView::new(request.output.buffer.as_ref(), &output_layout),
                },
                parameters,
            )
            .map_err(|source| WgpuBackendError::dispatch("transposed convolution", source))
    }

    fn convolution_transposed_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        backward::<T, R, D>(
            request,
            Parameters::Transposed(TransposedConvolutionParameters::new(
                stride,
                padding,
                output_padding,
                dilation,
            )),
        )
    }
}

enum Parameters<const D: usize> {
    Regular(leto::Result<ConvolutionParameters<D>>),
    Transposed(leto::Result<TransposedConvolutionParameters<D>>),
}

fn backward<T, const R: usize, const D: usize>(
    request: ConvolutionBackward<'_, WgpuBackend, T>,
    parameters: Parameters<D>,
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar + leto_ops::Scalar + DialectScalar<Wgsl>,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    let operation = match parameters {
        Parameters::Regular(_) => "convolution backward",
        Parameters::Transposed(_) => "transposed convolution backward",
    };
    let input_layout = layout::<R>(operation, request.input_layout)?;
    let weight_layout = layout::<R>(operation, request.weight_layout)?;
    let grad_output_layout = layout::<R>(operation, request.grad_output_layout)?;
    let grad_input_layout = request
        .grad_input
        .as_ref()
        .map(|_| layout::<R>(operation, request.grad_input_layout))
        .transpose()?;
    let grad_weight_layout = request
        .grad_weight
        .as_ref()
        .map(|_| layout::<R>(operation, request.grad_weight_layout))
        .transpose()?;
    let grad_bias_layout = request
        .grad_bias
        .as_ref()
        .map(|bias| bias_layout(operation, bias))
        .transpose()?;
    let operands = ConvolutionBackwardOperands {
        input: StridedView::new(request.input.buffer.as_ref(), &input_layout),
        weight: StridedView::new(request.weight.buffer.as_ref(), &weight_layout),
        grad_output: StridedView::new(request.grad_output.buffer.as_ref(), &grad_output_layout),
        gradients: ConvolutionGradientViews {
            input: request
                .grad_input
                .zip(grad_input_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer.as_ref(), layout)),
            weight: request
                .grad_weight
                .zip(grad_weight_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer.as_ref(), layout)),
            bias: request
                .grad_bias
                .zip(grad_bias_layout.as_ref())
                .map(|(buffer, layout)| StridedView::new(buffer.buffer.as_ref(), layout)),
        },
    };
    let operations = WgpuConvolutionOps;
    let result = match parameters {
        Parameters::Regular(parameters) => operations.convolution_backward_accumulate(
            &get_wgpu_context().hephaestus_device,
            operands,
            parameters.map_err(|error| {
                WgpuBackendError::Validation(BackendError::Storage {
                    operation,
                    reason: error.to_string(),
                })
            })?,
        ),
        Parameters::Transposed(parameters) => operations
            .convolution_transposed_backward_accumulate(
                &get_wgpu_context().hephaestus_device,
                operands,
                parameters.map_err(|error| {
                    WgpuBackendError::Validation(BackendError::Storage {
                        operation,
                        reason: error.to_string(),
                    })
                })?,
            ),
    };
    result.map_err(|source| WgpuBackendError::dispatch(operation, source))
}
