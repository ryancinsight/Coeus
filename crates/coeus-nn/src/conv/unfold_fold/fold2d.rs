use super::validation::{checked_output_dim, invalid_window};
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use std::marker::PhantomData;

/// Accumulates `[N, C*kH*kW, H_out*W_out]` back into `[N, C, output_h, output_w]`.
///
/// Inverse (adjoint) of [`super::Unfold2d`]. Overlapping contributions are summed.
/// Matches PyTorch `nn.Fold`.
#[derive(Clone, Debug)]
pub struct Fold2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    output_h: usize,
    output_w: usize,
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

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Fold2d<T, B> {
    /// Create `Fold2d` with the target output size and square kernel parameters.
    pub fn new(
        output_h: usize,
        output_w: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            output_h,
            output_w,
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
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Fold2d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "Fold2d",
                expected: "3",
                actual: shape.len(),
            });
        }
        let kernel_area = self
            .kernel_h
            .checked_mul(self.kernel_w)
            .filter(|&area| area != 0)
            .ok_or_else(|| {
                invalid_window(
                    "Fold2d",
                    "kernel dimensions",
                    vec![self.kernel_h, self.kernel_w],
                )
            })?;
        let output_rows = checked_output_dim(
            self.output_h,
            self.kernel_h,
            self.stride_h,
            self.padding_h,
            self.dilation_h,
        );
        let output_columns = checked_output_dim(
            self.output_w,
            self.kernel_w,
            self.stride_w,
            self.padding_w,
            self.dilation_w,
        );
        let expected_windows = output_rows
            .zip(output_columns)
            .and_then(|(rows, columns)| rows.checked_mul(columns))
            .filter(|&windows| windows != 0)
            .ok_or_else(|| {
                invalid_window(
                    "Fold2d",
                    "output shape and window configuration",
                    vec![
                        self.output_h,
                        self.output_w,
                        self.kernel_h,
                        self.kernel_w,
                        self.stride_h,
                        self.stride_w,
                        self.padding_h,
                        self.padding_w,
                        self.dilation_h,
                        self.dilation_w,
                    ],
                )
            })?;
        if !shape[1].is_multiple_of(kernel_area) || shape[2] != expected_windows {
            return Err(ModuleError::ShapeMismatch {
                module: "Fold2d",
                parameter: "folded channel and window dimensions",
                expected: vec![kernel_area, expected_windows],
                actual: vec![shape[1], shape[2]],
            });
        }

        coeus_autograd::fold2d(
            input,
            self.output_h,
            self.output_w,
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
            module: "Fold2d",
            source,
        })
    }
}
