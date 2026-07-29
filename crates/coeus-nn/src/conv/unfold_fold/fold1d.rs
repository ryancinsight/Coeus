use super::validation::{checked_output_dim, invalid_window};
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use std::marker::PhantomData;

/// Accumulates `[N, C*kernel_size, L_out]` back into `[N, C, output_size]`.
///
/// Inverse (adjoint) of [`super::Unfold1d`]. Overlapping window contributions are summed.
/// Matches PyTorch `nn.Fold` in 1D.
///
/// # Shape
/// - Input:  `[N, C * kernel_size, L_out]`
/// - Output: `[N, C, output_size]`
#[derive(Clone, Debug)]
pub struct Fold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    output_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Fold1d<T, B> {
    /// Create a `Fold1d` with the given hyperparameters.
    pub fn new(
        output_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Fold1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "Fold1d",
                expected: "3",
                actual: shape.len(),
            });
        }
        let expected_columns = checked_output_dim(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        .filter(|&output| output != 0)
        .ok_or_else(|| {
            invalid_window(
                "Fold1d",
                "output size, kernel, stride, padding, and dilation",
                vec![
                    self.output_size,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                ],
            )
        })?;
        if !shape[1].is_multiple_of(self.kernel_size) || shape[2] != expected_columns {
            return Err(ModuleError::ShapeMismatch {
                module: "Fold1d",
                parameter: "folded channel and column dimensions",
                expected: vec![self.kernel_size, expected_columns],
                actual: vec![shape[1], shape[2]],
            });
        }

        Ok(coeus_autograd::fold1d(
            input,
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        ))
    }
}
