//! Convolution sub-trait.
//!
//! [`ConvOps`] is the interface-segregated sub-trait for regular convolution
//! and 1-D/2-D transposed-convolution dispatch.  The 3-D transposed operation
//! lives in [`super::ConvTranspose3dOps`] so providers can own that capability
//! independently.

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

use super::super::defaults;

/// Convolution operations.
///
/// This sub-trait is one of seven concerns that compose
/// [`BackendOps`].  Backends implement `ConvOps` directly; the
/// blanket impl provides `BackendOps` automatically.
///
/// [`BackendOps`]: super::super::BackendOps
pub trait ConvOps<T: Scalar>: ComputeBackend {
    /// 1D Convolution
    fn conv1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    );

    /// 1D Convolution Backward
    fn conv1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    );

    /// 2D Convolution
    fn conv2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    );

    /// 2D Convolution Backward
    fn conv2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    );

    /// 3D Convolution
    fn conv3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    );

    /// 3D Convolution Backward
    fn conv3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    );

    /// 1-D Transposed Convolution default (host-side fallback).
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: Float,
    {
        defaults::conv_transpose::conv_transpose1d(
            self,
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            output,
            output_layout,
        )
    }

    /// 2-D Transposed Convolution default (host-side fallback).
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: Float,
    {
        defaults::conv_transpose::conv_transpose2d(
            self,
            input,
            input_layout,
            weight,
            weight_layout,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            output,
            output_layout,
        )
    }
}
