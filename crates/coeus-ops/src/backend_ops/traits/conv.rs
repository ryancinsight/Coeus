//! Fallible backend-owned convolution operation family.

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Borrowed operands for regular or transposed convolution forward dispatch.
pub struct ConvolutionForward<'a, B: ComputeBackend, T: Scalar> {
    /// Input activations.
    pub input: &'a B::DeviceBuffer<T>,
    /// Input layout.
    pub input_layout: &'a Layout,
    /// Kernel weights.
    pub weight: &'a B::DeviceBuffer<T>,
    /// Weight layout.
    pub weight_layout: &'a Layout,
    /// Optional channel bias.
    pub bias: Option<&'a B::DeviceBuffer<T>>,
    /// Caller-owned output.
    pub output: &'a mut B::DeviceBuffer<T>,
    /// Output layout.
    pub output_layout: &'a Layout,
}

/// Borrowed operands for additive convolution backward dispatch.
pub struct ConvolutionBackward<'a, B: ComputeBackend, T: Scalar> {
    /// Gradient of the forward output.
    pub grad_output: &'a B::DeviceBuffer<T>,
    /// Output-gradient layout.
    pub grad_output_layout: &'a Layout,
    /// Forward input activations.
    pub input: &'a B::DeviceBuffer<T>,
    /// Input layout.
    pub input_layout: &'a Layout,
    /// Forward kernel weights.
    pub weight: &'a B::DeviceBuffer<T>,
    /// Weight layout.
    pub weight_layout: &'a Layout,
    /// Optional input-gradient target.
    pub grad_input: Option<&'a mut B::DeviceBuffer<T>>,
    /// Input-gradient layout.
    pub grad_input_layout: &'a Layout,
    /// Optional weight-gradient target.
    pub grad_weight: Option<&'a mut B::DeviceBuffer<T>>,
    /// Weight-gradient layout.
    pub grad_weight_layout: &'a Layout,
    /// Optional contiguous bias-gradient target.
    pub grad_bias: Option<&'a mut B::DeviceBuffer<T>>,
}

/// Regular and transposed convolution operations.
///
/// Implementations own four rank-generic kernels. Rank-specific methods are
/// canonical defaults, so backend implementations cannot drift by rank.
pub trait ConvOps<T: Scalar>: ComputeBackend {
    /// Execute rank-generic regular convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    fn convolution_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>;

    /// Accumulate rank-generic regular-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    fn convolution_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>;

    /// Execute rank-generic transposed convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    fn convolution_transposed_forward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionForward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Accumulate rank-generic transposed-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    fn convolution_transposed_backward<const R: usize, const D: usize>(
        &self,
        request: ConvolutionBackward<'_, Self, T>,
        stride: [usize; D],
        padding: [usize; D],
        output_padding: [usize; D],
        dilation: [usize; D],
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Execute one-dimensional regular convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error> {
        self.convolution_forward::<3, 1>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride],
            [padding],
            [dilation],
        )
    }

    /// Accumulate one-dimensional regular-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv1d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
    ) -> Result<(), Self::Error> {
        self.convolution_backward::<3, 1>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride],
            [padding],
            [dilation],
        )
    }

    /// Execute two-dimensional regular convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error> {
        self.convolution_forward::<4, 2>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride; 2],
            [padding; 2],
            [dilation; 2],
        )
    }

    /// Accumulate two-dimensional regular-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv2d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
    ) -> Result<(), Self::Error> {
        self.convolution_backward::<4, 2>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride; 2],
            [padding; 2],
            [dilation; 2],
        )
    }

    /// Execute three-dimensional regular convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error> {
        self.convolution_forward::<5, 3>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride; 3],
            [padding; 3],
            [dilation; 3],
        )
    }

    /// Accumulate three-dimensional regular-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv3d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
    ) -> Result<(), Self::Error> {
        self.convolution_backward::<5, 3>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride; 3],
            [padding; 3],
            [dilation; 3],
        )
    }

    /// Execute one-dimensional transposed convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_forward::<3, 1>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride],
            [padding],
            [output_padding],
            [dilation],
        )
    }

    /// Accumulate one-dimensional transposed-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv_transpose1d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
        output_padding: usize,
        dilation: usize,
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_backward::<3, 1>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride],
            [padding],
            [output_padding],
            [dilation],
        )
    }

    /// Execute two-dimensional transposed convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
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
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_forward::<4, 2>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride; 2],
            [padding; 2],
            [output_padding; 2],
            [dilation; 2],
        )
    }

    /// Accumulate two-dimensional transposed-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv_transpose2d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
        output_padding: usize,
        dilation: usize,
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_backward::<4, 2>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride; 2],
            [padding; 2],
            [output_padding; 2],
            [dilation; 2],
        )
    }

    /// Execute three-dimensional transposed convolution.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv_transpose3d(
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
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_forward::<5, 3>(
            ConvolutionForward {
                input,
                input_layout,
                weight,
                weight_layout,
                bias,
                output,
                output_layout,
            },
            [stride; 3],
            [padding; 3],
            [output_padding; 3],
            [dilation; 3],
        )
    }

    /// Accumulate three-dimensional transposed-convolution gradients.
    ///
    /// # Errors
    ///
    /// Returns the backend error when validation or execution fails.
    #[expect(clippy::too_many_arguments, reason = "ratchet COEUS-LINT-1")]
    fn conv_transpose3d_backward(
        &self,
        grad_output: &Self::DeviceBuffer<T>,
        grad_output_layout: &Layout,
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
        output_padding: usize,
        dilation: usize,
    ) -> Result<(), Self::Error>
    where
        T: Float,
    {
        self.convolution_transposed_backward::<5, 3>(
            ConvolutionBackward {
                grad_output,
                grad_output_layout,
                input,
                input_layout,
                weight,
                weight_layout,
                grad_input,
                grad_input_layout,
                grad_weight,
                grad_weight_layout,
                grad_bias,
            },
            [stride; 3],
            [padding; 3],
            [output_padding; 3],
            [dilation; 3],
        )
    }
}
