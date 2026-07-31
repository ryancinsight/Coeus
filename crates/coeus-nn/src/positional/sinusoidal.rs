// ── Sinusoidal Positional Encoding ──
//
// PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
//
// The full `[max_len, d_model]` table is precomputed at construction.
// `forward` adds a non-owning slice view (zero-copy for CPU-addressable
// storage) to the input embedding tensor.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Sinusoidal (non-learnable) positional encoding layer.
///
/// Precomputes a `[max_len, d_model]` table and adds a row-slice of it to
/// the input at forward time.
pub struct SinusoidalEncoding<
    T: coeus_core::Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
> {
    /// Precomputed PE table: `[max_len, d_model]`.
    pub table: Tensor<T, B>,
    /// Maximum sequence length the precomputed table covers.
    pub max_len: usize,
    /// Embedding dimension (must be even).
    pub d_model: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> SinusoidalEncoding<T, B> {
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
        let mut table = Tensor::zeros_on([max_len, d_model], &backend);
        {
            use coeus_core::StorageMut;
            let data = table
                .storage_mut()
                .try_as_mut_slice()
                .expect("SinusoidalEncoding: backend must be CPU-addressable at construction");
            for pos in 0..max_len {
                for i in 0..(d_model / 2) {
                    let denom = 10_000.0_f64.powf(2.0 * i as f64 / d_model as f64);
                    let angle = pos as f64 / denom;
                    data[pos * d_model + 2 * i] = T::from_f64(angle.sin());
                    data[pos * d_model + 2 * i + 1] = T::from_f64(angle.cos());
                }
            }
        }
        Self {
            table,
            max_len,
            d_model,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for SinusoidalEncoding<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![] // non-learnable
    }

    /// Add positional encoding to `input`.
    ///
    /// - `input`: `[batch, seq_len, d_model]`
    ///
    /// Returns `[batch, seq_len, d_model]`.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let backend = B::default();
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

        // Extract the top `seq_len` rows of the PE table as a [seq_len, d_model] tensor.
        // For CPU-addressable storage this is a zero-copy slice; for GPU storage it
        // involves a staging copy limited to `seq_len * d_model` elements.
        let pe_slice = extract_pe_slice(&self.table, seq_len, self.d_model, &backend);
        let pe_var = Var::new(pe_slice, false);

        // Broadcast add: input [B, seq, d] + pe [seq, d] via autograd add.
        // autograd::add handles the broadcast accumulation.
        Ok(coeus_autograd::add(input, &pe_var))
    }
}

/// Extract the first `seq_len` rows from the PE table.
///
/// Returns a contiguous `[seq_len, d_model]` tensor.
fn extract_pe_slice<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    table: &Tensor<T, B>,
    seq_len: usize,
    d_model: usize,
    backend: &B,
) -> Tensor<T, B> {
    use coeus_core::{Storage, StorageMut};
    let mut out = Tensor::zeros_on([seq_len, d_model], backend);
    // Zero-copy path: both src and dst are CPU-addressable.
    if let (Some(src), Some(dst)) = (
        table.storage().try_as_slice(),
        out.storage_mut().try_as_mut_slice(),
    ) {
        dst.copy_from_slice(&src[..seq_len * d_model]);
    } else {
        // GPU path: stage through host.
        let total = table.numel();
        let mut host = vec![T::zero(); total];
        backend.copy_to_host(table.storage(), &mut host);
        let mut out_host = vec![T::zero(); seq_len * d_model];
        out_host.copy_from_slice(&host[..seq_len * d_model]);
        backend.copy_to_device(&out_host, out.storage_mut());
    }
    out
}
