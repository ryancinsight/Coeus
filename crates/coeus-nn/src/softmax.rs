// ── Softmax ──
// Softmax along a dimension with proper autograd backward pass.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::Float;

/// Functional softmax along `dim`.
pub fn softmax<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: isize,
) -> Var<T, B> {
    coeus_autograd::softmax(input, dim)
}

/// Softmax module (last dimension by default).
#[derive(Clone, Debug, Default)]
pub struct Softmax {
    /// Axis along which to apply softmax (negative indices count from the end).
    dim: isize,
}

impl Softmax {
    /// Create a `Softmax` module that normalizes along `dim`.
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softmax {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let rank = input.tensor.ndim();
        let requested = self.dim.unsigned_abs();
        let axis = if self.dim >= 0 {
            usize::try_from(self.dim).map_err(|_| ModuleError::InvalidAxis {
                module: "Softmax",
                axis: requested,
                rank,
            })?
        } else {
            rank.checked_sub(requested)
                .ok_or(ModuleError::InvalidAxis {
                    module: "Softmax",
                    axis: requested,
                    rank,
                })?
        };
        if axis >= rank {
            return Err(ModuleError::InvalidAxis {
                module: "Softmax",
                axis: requested,
                rank,
            });
        }
        let normalized = isize::try_from(axis).map_err(|_| ModuleError::InvalidAxis {
            module: "Softmax",
            axis,
            rank,
        })?;

        Ok(softmax(input, normalized))
    }
}
