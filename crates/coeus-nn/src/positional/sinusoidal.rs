// ── Sinusoidal Positional Encoding ──
//
// PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
//
// The full `[max_len, d_model]` table is precomputed at construction.
// `forward` adds a non-owning slice view over the selected backend allocation.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{ComputeBackend, Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Sinusoidal (non-learnable) positional encoding layer.
///
/// Precomputes a `[max_len, d_model]` table and adds a row-slice of it to
/// the input at forward time.
pub struct SinusoidalEncoding<T: coeus_core::Scalar, B: ComputeBackend + Default = MoiraiBackend> {
    /// Precomputed PE table: `[max_len, d_model]`.
    pub table: Tensor<T, B>,
    /// Maximum sequence length the precomputed table covers.
    pub max_len: usize,
    /// Embedding dimension (must be even).
    pub d_model: usize,
}

impl<T: Float, B: ComputeBackend + Default> SinusoidalEncoding<T, B> {
    /// Build the encoding table.
    ///
    /// - `max_len`: maximum sequence length supported.
    /// - `d_model`: embedding dimension (must be even).
    pub fn new(max_len: usize, d_model: usize) -> Self {
        assert!(
            d_model.is_multiple_of(2),
            "SinusoidalEncoding: d_model must be even, got {d_model}"
        );
        let backend = B::default();
        let table_len = max_len
            .checked_mul(d_model)
            .expect("SinusoidalEncoding: table element count overflows usize");
        let mut values = Vec::with_capacity(table_len);
        let base = T::from_usize(10_000);
        let dimension = T::from_usize(d_model);
        for pos in 0..max_len {
            let position = T::from_usize(pos);
            for i in 0..(d_model / 2) {
                let exponent = T::from_usize(
                    i.checked_mul(2)
                        .expect("SinusoidalEncoding: frequency index overflows usize"),
                ) / dimension;
                let angle = position / base.powf(exponent);
                values.push(angle.sin());
                values.push(angle.cos());
            }
        }
        let table = Tensor::from_slice_on([max_len, d_model], &values, &backend);
        Self {
            table,
            max_len,
            d_model,
        }
    }
}

impl<T: Float, B: coeus_ops::ElementwiseOps<T> + coeus_ops::ReductionOps<T> + Default> Module<T, B>
    for SinusoidalEncoding<T, B>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![] // non-learnable
    }

    /// Add positional encoding to `input`.
    ///
    /// - `input`: `[batch, seq_len, d_model]`
    ///
    /// Returns `[batch, seq_len, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape();
        if shape.len() != 3 {
            return Err(ModuleError::InvalidRank {
                module: "SinusoidalEncoding",
                expected: "3",
                actual: shape.len(),
            });
        }
        if shape[2] != self.d_model {
            return Err(ModuleError::ShapeMismatch {
                module: "SinusoidalEncoding",
                parameter: "input last dimension",
                expected: vec![self.d_model],
                actual: vec![shape[2]],
            });
        }
        let seq_len = shape[1];
        if seq_len > self.max_len {
            return Err(ModuleError::ShapeMismatch {
                module: "SinusoidalEncoding",
                parameter: "input sequence length",
                expected: vec![self.max_len],
                actual: vec![seq_len],
            });
        }

        // The view keeps the selected backend allocation. The elementwise
        // provider consumes the view layout directly, so forward never stages
        // or downloads the precomputed table.
        let pe_slice = prefix_view(&self.table, seq_len, self.d_model);
        let pe_var = Var::new(pe_slice, false);

        // Broadcast add: input [B, seq, d] + pe [seq, d] via autograd add.
        // autograd::add handles the broadcast accumulation.
        Ok(coeus_autograd::add(input, &pe_var))
    }
}

/// Borrow the active sequence prefix without changing its backend storage.
#[inline]
fn prefix_view<T: coeus_core::Scalar, B: ComputeBackend>(
    table: &Tensor<T, B>,
    seq_len: usize,
    d_model: usize,
) -> Tensor<T, B> {
    table.slice(&[(0, seq_len), (0, d_model)])
}

#[cfg(test)]
mod tests {
    use super::prefix_view;
    use coeus_core::{SequentialBackend, Storage};
    use coeus_tensor::Tensor;

    #[test]
    fn prefix_view_shares_cpu_storage() {
        let backend = SequentialBackend;
        let table = Tensor::from_slice_on([4, 6], &[0.0_f32; 24], &backend);
        let prefix = prefix_view(&table, 2, 6);

        let table_ptr = table
            .storage()
            .try_as_slice()
            .expect("SequentialBackend storage is CPU-addressable")
            .as_ptr();
        let prefix_ptr = prefix
            .storage()
            .try_as_slice()
            .expect("SequentialBackend storage is CPU-addressable")
            .as_ptr();
        assert_eq!(prefix_ptr, table_ptr);
        assert_eq!(prefix.shape(), &[2, 6]);
    }
}
