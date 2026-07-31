use super::dim::{ConvDim, ConvDispatch};
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;
use std::marker::PhantomData;

/// Generic convolution layer parameterised over spatial dimensionality `D`.
///
/// Use the type aliases [`crate::conv::Conv1d`], [`crate::conv::Conv2d`],
/// [`crate::conv::Conv3d`] for concrete dimensions rather than naming this type directly.
#[derive(Clone)]
pub struct Conv<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    D: ConvDim = super::dim::Dim2D,
> {
    /// Learned projection weight `[out_channels, in_channels, k...]`.
    pub weight: Var<T, B>,
    /// Optional learned bias `[out_channels]`.
    pub bias: Option<Var<T, B>>,
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Number of output feature channels.
    pub out_channels: usize,
    /// Isotropic kernel side length.
    pub kernel_size: usize,
    /// Isotropic convolution stride.
    pub stride: usize,
    /// Zero-padding applied symmetrically to each spatial side.
    pub padding: usize,
    /// Spacing between kernel elements (à trous / dilated convolution).
    pub dilation: usize,
    _dim: PhantomData<D>,
}

/// Dimension-independent convolution hyperparameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConvParams {
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Number of output feature channels.
    pub out_channels: usize,
    /// Isotropic kernel side length.
    pub kernel_size: usize,
    /// Isotropic convolution stride.
    pub stride: usize,
    /// Zero-padding applied symmetrically to each spatial side.
    pub padding: usize,
    /// Spacing between kernel elements (à trous / dilated convolution).
    pub dilation: usize,
}

impl ConvParams {
    /// Construct convolution parameters after validating positive stride and dilation.
    ///
    /// # Panics
    /// Panics if `stride == 0` or `dilation == 0`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        assert!(
            stride >= 1 && dilation >= 1,
            "stride and dilation must be >= 1"
        );
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default, D: ConvDim> Conv<T, B, D> {
    /// Construct with default stride=1, padding=0, dilation=1.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, bias: bool) -> Self {
        Self::with_params(in_channels, out_channels, kernel_size, 1, 0, 1, bias)
    }

    /// Construct with explicit hyperparameters.
    pub fn with_params(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self {
        let params = ConvParams::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        );
        let backend = B::default();
        let w_shape = D::weight_shape(params.out_channels, params.in_channels, params.kernel_size);
        let weight = Var::new(Tensor::ones_on(w_shape, &backend), true);
        let bias_var = if bias {
            Some(Var::new(
                Tensor::zeros_on([params.out_channels], &backend),
                true,
            ))
        } else {
            None
        };
        Self::from_vars(weight, bias_var, params)
    }

    /// Construct directly from existing weight and bias variables.
    ///
    /// Used by PyO3 bindings that build `Var`s from Python-supplied tensors.
    pub fn from_vars(weight: Var<T, B>, bias: Option<Var<T, B>>, params: ConvParams) -> Self {
        Self {
            weight,
            bias,
            in_channels: params.in_channels,
            out_channels: params.out_channels,
            kernel_size: params.kernel_size,
            stride: params.stride,
            padding: params.padding,
            dilation: params.dilation,
            _dim: PhantomData,
        }
    }

    #[inline]
    fn k_eff(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, D: ConvDim> Module<T, B> for Conv<T, B, D> {
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
        let expected_rank = D::SPATIAL_RANK + 2;
        if shape.len() != expected_rank {
            return Err(ModuleError::InvalidRank {
                module: "Conv",
                expected: match expected_rank {
                    3 => "3",
                    4 => "4",
                    5 => "5",
                    _ => "configured convolution rank",
                },
                actual: shape.len(),
            });
        }
        if shape[1] != self.in_channels {
            return Err(ModuleError::ChannelMismatch {
                module: "Conv",
                expected: self.in_channels,
                actual: shape[1],
            });
        }
        let in_spatial = &shape[2..];
        let k_eff = self.k_eff();
        let out_sp = D::out_spatial(in_spatial, k_eff, self.stride, self.padding);
        if out_sp.contains(&0) {
            return Err(ModuleError::ShapeMismatch {
                module: "Conv",
                parameter: "output spatial dimensions",
                expected: vec![1; D::SPATIAL_RANK],
                actual: out_sp,
            });
        }
        let out_shape = D::output_shape(shape[0], self.out_channels, &out_sp);
        let mut out_tensor = Tensor::zeros_on(out_shape, &backend);
        {
            let (out_storage, out_layout) = out_tensor.storage_mut_and_layout();
            D::backend_conv(ConvDispatch {
                backend: &backend,
                input_buf: input.tensor.storage(),
                input_layout: input.tensor.layout(),
                weight_buf: self.weight.tensor.storage(),
                weight_layout: self.weight.tensor.layout(),
                bias: self.bias.as_ref().map(|b| b.tensor.storage()),
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                out_buf: out_storage,
                out_layout,
            })
            .map_err(|source| ModuleError::Backend {
                module: "Conv",
                source,
            })?;
        }
        Ok(D::autograd_conv(
            input,
            &self.weight,
            &self.bias,
            out_tensor,
            self.stride,
            self.padding,
            self.dilation,
        ))
    }
}
