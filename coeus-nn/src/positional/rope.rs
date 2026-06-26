// ── Rotary Positional Embedding (RoPE) ──
//
// Applies a rotation to the input tensor (assumed shape [batch, seq_len, num_heads, d_head])
// using precomputed cos and sin tables.
//
// x_rotated = x * cos + rotate_half(x) * sin
//
// where rotate_half([x1, x2]) = [-x2, x1].

use crate::module::Module;
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

        let mut cos_table = Tensor::zeros_on([max_len, d_head], &backend);
        let mut sin_table = Tensor::zeros_on([max_len, d_head], &backend);

        {
            use coeus_core::StorageMut;
            let cos_slice = cos_table
                .storage_mut()
                .try_as_mut_slice()
                .expect("RotaryEmbedding: backend must be CPU-addressable at construction");
            let sin_slice = sin_table
                .storage_mut()
                .try_as_mut_slice()
                .expect("RotaryEmbedding: backend must be CPU-addressable at construction");

            let half_dim = d_head / 2;
            for pos in 0..max_len {
                for i in 0..half_dim {
                    let theta = base.powf(-2.0 * (i as f64) / (d_head as f64));
                    let angle = (pos as f64) * theta;
                    let c = T::from_f64(angle.cos());
                    let s = T::from_f64(angle.sin());

                    // Store duplicated (GPT-NeoX/LLaMA style)
                    cos_slice[pos * d_head + i] = c;
                    cos_slice[pos * d_head + i + half_dim] = c;

                    sin_slice[pos * d_head + i] = s;
                    sin_slice[pos * d_head + i + half_dim] = s;
                }
            }
        }

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
    pub fn forward(&self, x: &Var<T, B>) -> Var<T, B>
    where
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        let backend = B::default();
        let shape = x.tensor.shape();
        let ndim = shape.len();
        assert!(
            ndim >= 2,
            "RotaryEmbedding: input must have at least 2 dimensions"
        );
        let seq_len = shape[1];
        let d_head = shape[ndim - 1];
        assert_eq!(
            d_head, self.d_head,
            "RotaryEmbedding: input last dimension must match d_head"
        );

        // Extract the top `seq_len` rows from the PE tables.
        let cos_slice = extract_pe_slice(&self.cos, seq_len, self.d_head, &backend);
        let sin_slice = extract_pe_slice(&self.sin, seq_len, self.d_head, &backend);

        // Reshape cos and sin slices to [1, seq_len, 1, ..., d_head] for broadcasting
        let mut broadcast_shape = vec![1; ndim];
        broadcast_shape[1] = seq_len;
        broadcast_shape[ndim - 1] = d_head;

        let cos_var = Var::new(cos_slice.reshape(broadcast_shape.clone()), false);
        let sin_var = Var::new(sin_slice.reshape(broadcast_shape), false);

        // x_rot = x * cos + rotate_half(x) * sin
        let x_cos = coeus_autograd::mul(x, &cos_var);
        let rx = rotate_half(x);
        let rx_sin = coeus_autograd::mul(&rx, &sin_var);

        coeus_autograd::add(&x_cos, &rx_sin)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for RotaryEmbedding<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward(input)
    }
}

/// Helper function to extract the first `seq_len` rows from a table.
fn extract_pe_slice<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    table: &Tensor<T, B>,
    seq_len: usize,
    d_model: usize,
    backend: &B,
) -> Tensor<T, B> {
    use coeus_core::{Storage, StorageMut};
    let mut out = Tensor::zeros_on([seq_len, d_model], backend);
    if let (Some(src), Some(dst)) = (
        table.storage().try_as_slice(),
        out.storage_mut().try_as_mut_slice(),
    ) {
        dst.copy_from_slice(&src[..seq_len * d_model]);
    } else {
        let total = table.numel();
        let mut host = vec![T::zero(); total];
        backend.copy_to_host(table.storage(), &mut host);
        let mut out_host = vec![T::zero(); seq_len * d_model];
        out_host.copy_from_slice(&host[..seq_len * d_model]);
        backend.copy_to_device(&out_host, out.storage_mut());
    }
    out
}

/// Helper to compute rotate_half(x).
///
/// rotate_half([x1, x2]) = [-x2, x1]
fn rotate_half<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(x: &Var<T, B>) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let shape = x.tensor.shape();
    let ndim = shape.len();
    let d_head = shape[ndim - 1];
    let half = d_head / 2;

    // Split along the last dimension.
    let chunks = coeus_autograd::split(x, half, ndim - 1);
    assert_eq!(
        chunks.len(),
        2,
        "rotate_half: split should yield exactly 2 chunks"
    );

    let x1 = &chunks[0];
    let x2 = &chunks[1];

    let backend = B::default();
    let minus_one = Var::new(Tensor::full_on([1], T::from_f64(-1.0), &backend), false);
    let neg_x2 = coeus_autograd::mul(x2, &minus_one);

    coeus_autograd::cat(&[&neg_x2, x1], ndim - 1)
}
