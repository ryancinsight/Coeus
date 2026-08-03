// ── Rotary Positional Embedding (RoPE) ──
//
// Applies a rotation to the input tensor (assumed shape [batch, seq_len, num_heads, d_head])
// using precomputed cos and sin tables.
//
// x_rotated = x * cos + rotate_half(x) * sin
//
// where rotate_half([x1, x2]) = [-x2, x1].

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// Rotary Positional Embedding (RoPE) layer.
///
/// Precomputes `cos` and `sin` tables of shape `[max_len, d_head]`,
/// and applies them to query/key tensors of shape `[batch, seq_len, num_heads, d_head]`.
#[derive(Clone)]
pub struct RotaryEmbedding<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Precomputed cos table: `[max_len, d_head]`.
    pub cos: Tensor<T, B>,
    /// Precomputed sin table: `[max_len, d_head]`.
    pub sin: Tensor<T, B>,
    /// Maximum sequence length supported by the precomputed tables.
    pub max_len: usize,
    /// Dimension per attention head (must be even).
    pub d_head: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> RotaryEmbedding<T, B> {
    /// Create a new Rotary Embedding layer.
    ///
    /// - `max_len`: maximum sequence length.
    /// - `d_head`: dimension per attention head (must be even).
    /// - `base`: base value for theta (typically 10000.0).
    pub fn new(max_len: usize, d_head: usize, base: f64) -> Self {
        assert!(
            d_head.is_multiple_of(2),
            "RotaryEmbedding: d_head must be even, got {d_head}"
        );
        let backend = B::default();

        let mut cos_values = vec![T::zero(); max_len * d_head];
        let mut sin_values = vec![T::zero(); max_len * d_head];
        let half_dim = d_head / 2;
        for pos in 0..max_len {
            for i in 0..half_dim {
                let theta = base.powf(-2.0 * (i as f64) / (d_head as f64));
                let angle = (pos as f64) * theta;
                let c = T::from_f64(angle.cos());
                let s = T::from_f64(angle.sin());

                // GPT-NeoX/LLaMA stores each frequency in both half-vectors.
                cos_values[pos * d_head + i] = c;
                cos_values[pos * d_head + i + half_dim] = c;
                sin_values[pos * d_head + i] = s;
                sin_values[pos * d_head + i + half_dim] = s;
            }
        }
        let cos_table = Tensor::from_slice_on([max_len, d_head], &cos_values, &backend);
        let sin_table = Tensor::from_slice_on([max_len, d_head], &sin_values, &backend);

        Self {
            cos: cos_table,
            sin: sin_table,
            max_len,
            d_head,
        }
    }

    /// Forward pass applying RoPE to `x`.
    ///
    /// `x` is assumed to have shape `[batch, seq_len, num_heads, d_head]` or any shape
    /// where dimension 1 is `seq_len` and the last dimension is `d_head`.
    pub fn forward(&self, x: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>>
    where
        B: coeus_ops::RotateHalfOps<T>,
    {
        let shape = x.tensor.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(ModuleError::InvalidRank {
                module: "RotaryEmbedding",
                expected: "at least 2",
                actual: ndim,
            });
        }
        let seq_len = shape[1];
        let d_head = shape[ndim - 1];
        if d_head != self.d_head {
            return Err(ModuleError::ShapeMismatch {
                module: "RotaryEmbedding",
                parameter: "input last dimension",
                expected: vec![self.d_head],
                actual: vec![d_head],
            });
        }
        if seq_len > self.max_len {
            return Err(ModuleError::ShapeMismatch {
                module: "RotaryEmbedding",
                parameter: "input sequence length",
                expected: vec![self.max_len],
                actual: vec![seq_len],
            });
        }

        // Extract the top `seq_len` rows from the PE tables.
        let cos_slice = extract_pe_slice(&self.cos, seq_len, self.d_head);
        let sin_slice = extract_pe_slice(&self.sin, seq_len, self.d_head);

        // Reshape cos and sin slices to [1, seq_len, 1, ..., d_head] for broadcasting
        let mut broadcast_shape = vec![1; ndim];
        broadcast_shape[1] = seq_len;
        broadcast_shape[ndim - 1] = d_head;

        let cos_var = Var::new(cos_slice.reshape(broadcast_shape.clone()), false);
        let sin_var = Var::new(sin_slice.reshape(broadcast_shape), false);

        // x_rot = x * cos + rotate_half(x) * sin
        let x_cos = coeus_autograd::mul(x, &cos_var);
        let rx = coeus_autograd::rotate_half(x).map_err(|source| ModuleError::Backend {
            module: "RotaryEmbedding",
            source,
        })?;
        let rx_sin = coeus_autograd::mul(&rx, &sin_var);

        Ok(coeus_autograd::add(&x_cos, &rx_sin))
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for RotaryEmbedding<T, B>
where
    B: coeus_ops::RotateHalfOps<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        self.forward(input)
    }
}

/// Helper function to extract the first `seq_len` rows from a table.
fn extract_pe_slice<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    table: &Tensor<T, B>,
    seq_len: usize,
    d_model: usize,
) -> Tensor<T, B> {
    table.slice(&[(0, seq_len), (0, d_model)])
}
