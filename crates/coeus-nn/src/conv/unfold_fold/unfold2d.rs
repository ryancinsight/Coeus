use super::validation::{checked_output_dim, invalid_window};
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use std::marker::PhantomData;

/// Extracts sliding 2D windows from `[N, C, H, W]` into `[N, C*kH*kW, H_out*W_out]`.
///
/// Matches PyTorch `nn.Unfold`. Stateless; no learnable parameters.
///
/// # Shape
/// - Input:  `[N, C, H, W]`
/// - Output: `[N, C * kH * kW, H_out * W_out]`
#[derive(Clone, Debug)]
pub struct Unfold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Unfold2d<T, B> {
    /// Create `Unfold2d` with a square kernel and equal per-axis parameters.
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_h: kernel_size,
            kernel_w: kernel_size,
            stride_h: stride,
            stride_w: stride,
            padding_h: padding,
            padding_w: padding,
            dilation_h: dilation,
            dilation_w: dilation,
            _marker: PhantomData,
        }
    }

    /// Create `Unfold2d` with per-axis hyperparameters.
    #[expect(
        clippy::too_many_arguments,
        reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
    )]
    pub fn with_params(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Unfold2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape();
        if shape.len() != 4 {
            return Err(ModuleError::InvalidRank {
                module: "Unfold2d",
                expected: "4",
                actual: shape.len(),
            });
        }
        let output_h = checked_output_dim(
            shape[2],
            self.kernel_h,
            self.stride_h,
            self.padding_h,
            self.dilation_h,
        );
        let output_w = checked_output_dim(
            shape[3],
            self.kernel_w,
            self.stride_w,
            self.padding_w,
            self.dilation_w,
        );
        let (output_h, output_w) = match (output_h, output_w) {
            (Some(output_h), Some(output_w)) if output_h != 0 && output_w != 0 => {
                (output_h, output_w)
            }
            _ => {
                return Err(invalid_window(
                    "Unfold2d",
                    "spatial shape and window configuration",
                    vec![
                        shape[2],
                        shape[3],
                        self.kernel_h,
                        self.kernel_w,
                        self.stride_h,
                        self.stride_w,
                        self.padding_h,
                        self.padding_w,
                        self.dilation_h,
                        self.dilation_w,
                    ],
                ));
            }
        };
        shape[1]
            .checked_mul(self.kernel_h)
            .and_then(|channels| channels.checked_mul(self.kernel_w))
            .and_then(|channels| channels.checked_mul(output_h))
            .and_then(|elements| elements.checked_mul(output_w))
            .ok_or_else(|| {
                invalid_window(
                    "Unfold2d",
                    "output element count",
                    vec![shape[1], self.kernel_h, self.kernel_w, output_h, output_w],
                )
            })?;

        coeus_autograd::unfold2d(
            input,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )
        .map_err(|source| ModuleError::Backend {
            module: "Unfold2d",
            source,
        })
    }
}
