//! Generic N-dimensional transposed (fractional-stride) convolution.
//!
//! [`ConvTranspose`] is the single implementation behind the
//! `ConvTranspose1d`/`ConvTranspose2d`/`ConvTranspose3d` aliases. The spatial
//! rank is a const parameter rather than a `ConvDim` marker: const generics
//! need no `PhantomData`, so `ConvTranspose1d::<f64, B> { .. }` struct-literal
//! construction keeps working unchanged, and it agrees with the `const DIM`
//! idiom already used by `coeus-autograd`'s batch-norm nodes.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// Transposed-convolution weight shape `[C_in, C_out, k...]`.
fn weight_shape<const DIM: usize>(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Vec<usize> {
    let mut shape = Vec::with_capacity(2 + DIM);
    shape.push(in_channels);
    shape.push(out_channels);
    shape.extend(std::iter::repeat_n(kernel_size, DIM));
    shape
}

/// Diagnostic module name for a given spatial rank.
const fn module_name<const DIM: usize>() -> &'static str {
    match DIM {
        1 => "ConvTranspose1d",
        2 => "ConvTranspose2d",
        3 => "ConvTranspose3d",
        _ => "ConvTransposeNd",
    }
}

/// Expected-rank description for the rank check.
const fn rank_str<const DIM: usize>() -> &'static str {
    match DIM {
        1 => "3",
        2 => "4",
        3 => "5",
        _ => "configured transposed-convolution rank",
    }
}

/// N-dimensional Transposed Convolution parameterised over spatial rank `DIM`.
///
/// Weight convention: `[C_in, C_out, k...]` (groups=1; the in/out channel order
/// is reversed relative to the regular [`crate::conv::Conv`]). Prefer the
/// `ConvTranspose1d`/`ConvTranspose2d`/`ConvTranspose3d` aliases.
#[derive(Clone)]
pub struct ConvTranspose<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const DIM: usize = 1,
> {
    /// Transposed convolution weight: `[in_channels, out_channels, k...]`.
    pub weight: Var<T, B>,
    /// Optional bias: `[out_channels]`.
    pub bias: Option<Var<T, B>>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Isotropic kernel side length.
    pub kernel_size: usize,
    /// Stride (upsampling factor).
    pub stride: usize,
    /// Input-side zero-padding removed from the output.
    pub padding: usize,
    /// Extra output size added to one side to resolve output-size ambiguity.
    pub output_padding: usize,
    /// Spacing between kernel elements.
    pub dilation: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> ConvTranspose<T, B, DIM> {
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
        T: coeus_leto::RealScalar,
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
        T: coeus_leto::RealScalar,
        B: coeus_ops::RandomInitOps<T>,
    {
        let backend = B::default();
        // Weight: [C_in, C_out, k...] — transposed convention.
        let w_shape = weight_shape::<DIM>(in_channels, out_channels, kernel_size);
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
}

// ── Rank-specific output-size helpers ──
//
// One inherent impl per rank keeps each alias's method surface exactly what it
// was before the merge (`output_len` for 1-D; `output_dims` for 2-D and 3-D).

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> ConvTranspose<T, B, 1> {
    /// Compute the output length for a given input length.
    pub fn output_len(&self, l: usize) -> usize {
        coeus_ops::conv_transpose::conv_transpose1d_output_len(
            l,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> ConvTranspose<T, B, 2> {
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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> ConvTranspose<T, B, 3> {
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

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const DIM: usize> Module<T, B>
    for ConvTranspose<T, B, DIM>
where
    T: coeus_leto::RealScalar,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let module: &str = module_name::<DIM>();
        let backend = B::default();
        let shape = input.tensor.shape();
        if shape.len() != DIM + 2 {
            return Err(ModuleError::InvalidRank {
                module,
                expected: rank_str::<DIM>(),
                actual: shape.len(),
            });
        }
        if shape[1] != self.in_channels {
            return Err(ModuleError::ChannelMismatch {
                module,
                expected: self.in_channels,
                actual: shape[1],
            });
        }
        let n = shape[0];
        let out_spatial: Vec<usize> = match DIM {
            1 => vec![coeus_ops::conv_transpose::conv_transpose1d_output_len(
                shape[2],
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            )],
            2 => {
                let (h_out, w_out) = coeus_ops::conv_transpose::conv_transpose2d_output_dims(
                    shape[2],
                    shape[3],
                    self.kernel_size,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                );
                vec![h_out, w_out]
            }
            3 => {
                let (d_out, h_out, w_out) = coeus_ops::conv_transpose::conv_transpose3d_output_dims(
                    shape[2],
                    shape[3],
                    shape[4],
                    self.kernel_size,
                    self.kernel_size,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                );
                vec![d_out, h_out, w_out]
            }
            _ => panic!("ConvTranspose: unsupported DIM {DIM}"),
        };
        let mut out_shape = Vec::with_capacity(2 + DIM);
        out_shape.push(n);
        out_shape.push(self.out_channels);
        out_shape.extend_from_slice(&out_spatial);

        let mut out_tensor = Tensor::zeros_on(out_shape, &backend);
        let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
        let dispatch = match DIM {
            1 => backend.conv_transpose1d(
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
            ),
            2 => backend.conv_transpose2d(
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
            ),
            3 => backend.conv_transpose3d(
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
            ),
            _ => panic!("ConvTranspose: unsupported DIM {DIM}"),
        };
        dispatch.map_err(|source| ModuleError::Backend { module, source })?;

        Ok(match DIM {
            1 => coeus_autograd::conv_transpose1d(
                input,
                &self.weight,
                &self.bias,
                out_tensor,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            ),
            2 => coeus_autograd::conv_transpose2d(
                input,
                &self.weight,
                &self.bias,
                out_tensor,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            ),
            3 => coeus_autograd::conv_transpose3d(
                input,
                &self.weight,
                &self.bias,
                out_tensor,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            ),
            _ => panic!("ConvTranspose: unsupported DIM {DIM}"),
        })
    }
}
