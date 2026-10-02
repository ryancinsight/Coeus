//! Zero-copy Coeus-to-Leto convolution dispatch.

mod regular;
mod transposed;
mod views;

use coeus_core::Layout;

pub use regular::{convolution_backward_accumulate, convolution_forward_into};
pub use transposed::{
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
};

/// Borrowed read-only Coeus operand.
#[derive(Clone, Copy)]
pub struct ReadOperand<'a, T> {
    /// Logical shape, strides, and storage offset.
    pub layout: &'a Layout,
    /// Borrowed CPU storage.
    pub data: &'a [T],
}

/// Borrowed mutable Coeus operand.
pub struct WriteOperand<'a, T> {
    /// Logical shape, strides, and storage offset.
    pub layout: &'a Layout,
    /// Exclusively borrowed CPU storage.
    pub data: &'a mut [T],
}

/// Forward convolution operands.
pub struct ConvolutionForward<'a, T> {
    /// Input activations.
    pub input: ReadOperand<'a, T>,
    /// Convolution weights.
    pub weight: ReadOperand<'a, T>,
    /// Optional contiguous output-channel bias.
    pub bias: Option<&'a [T]>,
    /// Caller-owned output.
    pub output: WriteOperand<'a, T>,
}

/// Selected additive convolution gradient targets.
pub struct ConvolutionGradients<'a, T> {
    /// Optional input-gradient target.
    pub input: Option<WriteOperand<'a, T>>,
    /// Optional weight-gradient target.
    pub weight: Option<WriteOperand<'a, T>>,
    /// Optional contiguous output-channel bias-gradient target.
    pub bias: Option<&'a mut [T]>,
}

/// Backward convolution operands.
pub struct ConvolutionBackward<'a, T> {
    /// Input activations from the forward pass.
    pub input: ReadOperand<'a, T>,
    /// Convolution weights.
    pub weight: ReadOperand<'a, T>,
    /// Gradient of the forward output.
    pub grad_output: ReadOperand<'a, T>,
    /// Selected additive gradient targets.
    pub gradients: ConvolutionGradients<'a, T>,
}

pub(super) fn bias_layout(channels: usize) -> leto::Result<leto::Layout<1>> {
    leto::Layout::c_contiguous([channels])
}
