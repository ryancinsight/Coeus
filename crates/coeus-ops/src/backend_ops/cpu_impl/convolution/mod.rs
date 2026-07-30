//! Direct zero-copy CPU convolution dispatch through Leto.

use coeus_core::{
    BackendError, ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar,
};
use leto::{ConvolutionParameters, TransposedConvolutionParameters};
use leto_ops::Scalar as LetoScalar;

use super::error::map_leto_error;
use coeus_leto::{
    ConvolutionBackward, ConvolutionForward, ConvolutionGradients, ReadOperand, WriteOperand,
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
};

pub(super) struct Forward<'a, B: ComputeBackend, T: Scalar> {
    pub input: &'a B::DeviceBuffer<T>,
    pub input_layout: &'a Layout,
    pub weight: &'a B::DeviceBuffer<T>,
    pub weight_layout: &'a Layout,
    pub bias: Option<&'a B::DeviceBuffer<T>>,
    pub output: &'a mut B::DeviceBuffer<T>,
    pub output_layout: &'a Layout,
}

pub(super) struct Backward<'a, B: ComputeBackend, T: Scalar> {
    pub grad_output: &'a B::DeviceBuffer<T>,
    pub grad_output_layout: &'a Layout,
    pub input: &'a B::DeviceBuffer<T>,
    pub input_layout: &'a Layout,
    pub weight: &'a B::DeviceBuffer<T>,
    pub weight_layout: &'a Layout,
    pub grad_input: Option<&'a mut B::DeviceBuffer<T>>,
    pub grad_input_layout: &'a Layout,
    pub grad_weight: Option<&'a mut B::DeviceBuffer<T>>,
    pub grad_weight_layout: &'a Layout,
    pub grad_bias: Option<&'a mut B::DeviceBuffer<T>>,
}

fn forward_operands<'a, B, T>(request: Forward<'a, B, T>) -> ConvolutionForward<'a, T>
where
    B: ComputeBackend,
    T: Scalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    ConvolutionForward {
        input: ReadOperand {
            layout: request.input_layout,
            data: request.input.as_slice(),
        },
        weight: ReadOperand {
            layout: request.weight_layout,
            data: request.weight.as_slice(),
        },
        bias: request.bias.map(CpuAddressableStorage::as_slice),
        output: WriteOperand {
            layout: request.output_layout,
            data: request.output.as_mut_slice(),
        },
    }
}

fn backward_operands<'a, B, T>(request: Backward<'a, B, T>) -> ConvolutionBackward<'a, T>
where
    B: ComputeBackend,
    T: Scalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    ConvolutionBackward {
        input: ReadOperand {
            layout: request.input_layout,
            data: request.input.as_slice(),
        },
        weight: ReadOperand {
            layout: request.weight_layout,
            data: request.weight.as_slice(),
        },
        grad_output: ReadOperand {
            layout: request.grad_output_layout,
            data: request.grad_output.as_slice(),
        },
        gradients: ConvolutionGradients {
            input: request.grad_input.map(|buffer| WriteOperand {
                layout: request.grad_input_layout,
                data: buffer.as_mut_slice(),
            }),
            weight: request.grad_weight.map(|buffer| WriteOperand {
                layout: request.grad_weight_layout,
                data: buffer.as_mut_slice(),
            }),
            bias: request
                .grad_bias
                .map(CpuAddressableStorageMut::as_mut_slice),
        },
    }
}

pub(super) fn regular_forward<B, T, const R: usize, const D: usize>(
    request: Forward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), BackendError>
where
    B: ComputeBackend,
    T: Scalar + LetoScalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let parameters = ConvolutionParameters::new(stride, padding, dilation)
        .map_err(|error| map_leto_error("convolution", error))?;
    convolution_forward_into::<T, R, D>(forward_operands(request), parameters)
        .map_err(|error| map_leto_error("convolution", error))
}

pub(super) fn regular_backward<B, T, const R: usize, const D: usize>(
    request: Backward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), BackendError>
where
    B: ComputeBackend,
    T: Scalar + LetoScalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let parameters = ConvolutionParameters::new(stride, padding, dilation)
        .map_err(|error| map_leto_error("convolution backward", error))?;
    convolution_backward_accumulate::<T, R, D>(backward_operands(request), parameters)
        .map_err(|error| map_leto_error("convolution backward", error))
}

pub(super) fn transposed_forward<B, T, const R: usize, const D: usize>(
    request: Forward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), BackendError>
where
    B: ComputeBackend,
    T: Scalar + LetoScalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let parameters =
        TransposedConvolutionParameters::new(stride, padding, output_padding, dilation)
            .map_err(|error| map_leto_error("transposed convolution", error))?;
    convolution_transposed_forward_into::<T, R, D>(forward_operands(request), parameters)
        .map_err(|error| map_leto_error("transposed convolution", error))
}

pub(super) fn transposed_backward<B, T, const R: usize, const D: usize>(
    request: Backward<'_, B, T>,
    stride: [usize; D],
    padding: [usize; D],
    output_padding: [usize; D],
    dilation: [usize; D],
) -> Result<(), BackendError>
where
    B: ComputeBackend,
    T: Scalar + LetoScalar,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    let parameters =
        TransposedConvolutionParameters::new(stride, padding, output_padding, dilation)
            .map_err(|error| map_leto_error("transposed convolution backward", error))?;
    convolution_transposed_backward_accumulate::<T, R, D>(backward_operands(request), parameters)
        .map_err(|error| map_leto_error("transposed convolution backward", error))
}
