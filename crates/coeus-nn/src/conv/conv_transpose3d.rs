// ── ConvTranspose3d ──
//
// 3-D Transposed (Fractional-Stride) Convolution, matching
// `torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size, ...)`.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// 3-D Transposed Convolution layer.
///
/// Weight convention: `[C_in, C_out, KD, KH, KW]` (groups=1; the in/out
/// channel order is reversed relative to the regular Conv3d).
/// Dispatch follows the selected backend's provider-owned convolution kernel.
#[derive(Clone)]
pub struct ConvTranspose3d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Transposed convolution weight: `[in_channels, out_channels, kD, kH, kW]`.
    pub weight: Var<T, B>,
    /// Optional bias: `[out_channels]`.
    pub bias: Option<Var<T, B>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Cubic kernel side length.
    pub kernel_size: usize,
    /// Stride (upsampling factor) along D, H, and W.
    pub stride: usize,
    /// Input-side zero-padding removed from the output.
    pub padding: usize,
    /// Extra output size added to one side to resolve output-size ambiguity.
    pub output_padding: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
}

impl<T: Scalar + coeus_core::Float, B: coeus_ops::BackendOps<T> + Default> ConvTranspose3d<T, B> {
    /// Create with default stride=1, padding=0, output_padding=0, dilation=1.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the fan is invalid or the selected backend
    /// cannot initialize the weight.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        T: coeus_leto::RandomScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        Self::with_params(in_channels, out_channels, kernel_size, 1, 0, 0, 1, bias)
    }

    /// Create with explicit hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns a typed error when the fan is invalid or the selected backend
    /// cannot initialize the weight.
    #[expect(clippy::too_many_arguments, reason = "transposed convolution contract")]
    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        T: coeus_leto::RandomScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        let backend = B::default();
        // Weight: [C_in, C_out, K, K, K] — transposed convention.
        let w_shape = [
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        ];
        let w_tensor = Tensor::ones_on(w_shape, &backend);
        let mut weight = Var::new(w_tensor, true);
        crate::init::kaiming_uniform(&mut weight, in_channels)?;
        let bias_var = if bias {
            Some(Var::new(Tensor::zeros_on([out_channels], &backend), true))
        } else {
            None
        };
        Ok(Self {
            weight,
            bias: bias_var,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        })
    }

    /// Compute the output spatial dimensions for a given input shape.
    pub fn output_dims(&self, d: usize, h: usize, w: usize) -> (usize, usize, usize) {
        coeus_ops::conv_transpose::conv_transpose3d_output_dims(
            d,
            h,
            w,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ConvTranspose3d<T, B>
where
    T: coeus_leto::RandomScalar,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != 5 {
            return Err(ModuleError::InvalidRank {
                module: "ConvTranspose3d",
                expected: "5",
                actual: shape.len(),
            });
        }
        if shape[1] != self.in_channels {
            return Err(ModuleError::ChannelMismatch {
                module: "ConvTranspose3d",
                expected: self.in_channels,
                actual: shape[1],
            });
        }
        let n = shape[0];
        let d = shape[2];
        let h = shape[3];
        let w = shape[4];
        let (d_out, h_out, w_out) = self.output_dims(d, h, w);

        let mut out_tensor =
            Tensor::zeros_on([n, self.out_channels, d_out, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
        backend
            .conv_transpose3d(
                input.tensor.storage(),
                input.tensor.layout(),
                self.weight.tensor.storage(),
                self.weight.tensor.layout(),
                self.bias.as_ref().map(|b| b.tensor.storage()),
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                out_storage,
                out_layout,
            )
            .map_err(|source| ModuleError::Backend {
                module: "ConvTranspose3d",
                source,
            })?;

        Ok(coeus_autograd::conv_transpose3d(
            input,
            &self.weight,
            &self.bias,
            out_tensor,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        ))
    }
}
