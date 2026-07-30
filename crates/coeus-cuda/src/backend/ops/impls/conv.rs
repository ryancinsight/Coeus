use crate::backend::{get_cuda_device, CudaBackend, CudaScalar};
use coeus_core::{BackendError, Float};
use coeus_ops::{ConvolutionBackward, ConvolutionForward};
use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionGradientViews,
    ConvolutionOps, DeviceBuffer, StridedView,
};
use hephaestus_cuda::{CudaConvolutionOps, CudaDevice};
use leto::{ConvolutionParameters, Layout, TransposedConvolutionParameters};

fn layout<const R: usize>(
    operation: &'static str,
    source: &coeus_core::Layout,
) -> Result<Layout<R>, crate::CudaBackendError> {
    coeus_leto::to_leto_layout(source).map_err(|error| crate::CudaBackendError::Validation {
        source: BackendError::Storage {
            operation,
            reason: error.to_string(),
        },
    })
}

fn bias_layout<T: CudaScalar>(
    operation: &'static str,
    storage: &crate::CudaStorage<T>,
) -> Result<Layout<1>, crate::CudaBackendError> {
    Layout::c_contiguous([storage.buffer.len()]).map_err(|error| {
        crate::CudaBackendError::Validation {
            source: BackendError::Storage {
                operation,
                reason: error.to_string(),
            },
        }
    })
}

impl<T> coeus_ops::ConvOps<T> for CudaBackend
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
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
                crate::CudaBackendError::Validation {
                    source: BackendError::Storage {
                        operation: "convolution",
                        reason: error.to_string(),
                    },
                }
            })?;
        CudaConvolutionOps
            .convolution_forward_into(
                get_cuda_device(),
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
            .map_err(|source| crate::CudaBackendError::dispatch("convolution", source))
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
                .map_err(|error| crate::CudaBackendError::Validation {
                    source: BackendError::Storage {
                        operation: "transposed convolution",
                        reason: error.to_string(),
                    },
                })?;
        CudaConvolutionOps
            .convolution_transposed_forward_into(
                get_cuda_device(),
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
            .map_err(|source| crate::CudaBackendError::dispatch("transposed convolution", source))
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
    request: ConvolutionBackward<'_, CudaBackend, T>,
    parameters: Parameters<D>,
) -> Result<(), crate::CudaBackendError>
where
    T: CudaScalar,
    CudaConvolutionOps: ConvolutionOps<CudaDevice, T>,
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
    let result = match parameters {
        Parameters::Regular(parameters) => CudaConvolutionOps.convolution_backward_accumulate(
            get_cuda_device(),
            operands,
            parameters.map_err(|error| crate::CudaBackendError::Validation {
                source: BackendError::Storage {
                    operation,
                    reason: error.to_string(),
                },
            })?,
        ),
        Parameters::Transposed(parameters) => CudaConvolutionOps
            .convolution_transposed_backward_accumulate(
                get_cuda_device(),
                operands,
                parameters.map_err(|error| crate::CudaBackendError::Validation {
                    source: BackendError::Storage {
                        operation,
                        reason: error.to_string(),
                    },
                })?,
            ),
    };
    result.map_err(|source| crate::CudaBackendError::dispatch(operation, source))
}
