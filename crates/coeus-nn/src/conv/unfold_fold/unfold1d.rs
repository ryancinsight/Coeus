use super::validation::{checked_output_dim, invalid_window};
use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use std::marker::PhantomData;

/// Extracts sliding windows from `[N, C, L]` into `[N, C*kernel_size, L_out]`.
///
/// Stateless module with no learnable parameters. Matches PyTorch `nn.Unfold` in 1D.
///
/// # Shape
/// - Input:  `[N, C, L]`
/// - Output: `[N, C * kernel_size, L_out]`
///   where `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Examples
///
/// ```
/// use coeus_nn::{Unfold1d, Module};
/// use coeus_autograd::Var;
/// use coeus_tensor::Tensor;
/// use coeus_core::SequentialBackend;
///
/// let m = Unfold1d::<f32, SequentialBackend>::new(3, 1, 0, 1);
/// let x = Var::new(Tensor::<f32, SequentialBackend>::ones([1, 2, 5]), false);
/// let y = m.forward(&x).expect("valid Unfold1d input");
/// assert_eq!(y.tensor.shape(), &[1, 6, 3]);
/// ```
#[derive(Clone, Debug)]
pub struct Unfold1d<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    _marker: PhantomData<(T, B)>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Unfold1d<T, B> {
    /// Create an `Unfold1d` with the given hyperparameters.
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Unfold1d<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "Unfold1d",
                expected: "3",
                actual: shape.len(),
            });
        }
        let output = checked_output_dim(
            shape[2],
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        .filter(|&output| output != 0)
        .ok_or_else(|| {
            invalid_window(
                "Unfold1d",
                "kernel, stride, padding, and dilation",
                vec![
                    shape[2],
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                ],
            )
        })?;
        shape[1]
            .checked_mul(self.kernel_size)
            .and_then(|channels| channels.checked_mul(output))
            .ok_or_else(|| {
                invalid_window(
                    "Unfold1d",
                    "output element count",
                    vec![shape[1], self.kernel_size, output],
                )
            })?;

        coeus_autograd::unfold1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        .map_err(|source| ModuleError::Backend {
            module: "Unfold1d",
            source,
        })
    }
}
