// ── ConvTranspose2d ──
//
// 2-D Transposed (Fractional-Stride) Convolution, matching
// `torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, ...)`.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// 2-D Transposed Convolution layer.
///
/// Weight convention: `[C_in, C_out, KH, KW]` (groups=1; opposite of Conv2d).
#[derive(Clone)]
pub struct ConvTranspose2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub weight: Var<T, B>,
    pub bias: Option<Var<T, B>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub dilation: usize,
}

impl<T: Scalar + coeus_core::Float, B: coeus_ops::BackendOps<T> + Default> ConvTranspose2d<T, B> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, bias: bool) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        Self::with_params(in_channels, out_channels, kernel_size, 1, 0, 0, 1, bias)
    }

    /// Create with explicit hyperparameters.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        let backend = B::default();
        // Weight: [C_in, C_out, KH, KW] — transposed convention.
        let w_shape = [in_channels, out_channels, kernel_size, kernel_size];
        let w_tensor = Tensor::ones_on(w_shape, &backend);
        let mut weight = Var::new(w_tensor, true);
        crate::init::kaiming_uniform(&mut weight, in_channels);
        let bias_var = if bias {
            Some(Var::new(Tensor::zeros_on([out_channels], &backend), true))
        } else {
            None
        };
        Self {
            weight,
            bias: bias_var,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
        }
    }

    /// Compute the output spatial dimensions for a given input shape.
    pub fn output_dims(&self, h: usize, w: usize) -> (usize, usize) {
        coeus_ops::conv_transpose::conv_transpose2d_output_dims(
            h,
            w,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ConvTranspose2d<T, B>
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

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let backend = B::default();
        let n = input.tensor.shape()[0];
        let h = input.tensor.shape()[2];
        let w = input.tensor.shape()[3];
        let (h_out, w_out) = self.output_dims(h, w);

        let mut out_tensor = Tensor::zeros_on([n, self.out_channels, h_out, w_out], &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
        backend.conv_transpose2d(
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
        );
        Var::new(out_tensor, false)
    }
}
