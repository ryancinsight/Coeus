// ── Multi-Head Attention ──
//
// Uses separate W_Q, W_K, W_V, W_O weight matrices (no fused QKV) so that
// projection is a generic tracked matmul with no CpuAddressableStorageMut requirement.
// Head split/merge use coeus_autograd::reshape (tracked) to preserve gradient flow.

use crate::init::kaiming_uniform;
use crate::module::Module;
use coeus_autograd::{AttentionMask, Var};
use coeus_core::{Float, MoiraiBackend, Scalar};
use std::marker::PhantomData;

/// Multi-head self/cross-attention.
///
/// # Type parameters
/// - `H` — number of attention heads (const generic); `d_model % H == 0` is asserted at construction.
/// - `M` — masking strategy ZST from `coeus_autograd` (`NullMask` or `CausalMask`)
///
/// Uses four separate projection weights W_Q, W_K, W_V, W_O, each shape `[d_model, d_model]`,
/// projected via tracked matmul. Head reshape uses `coeus_autograd::reshape` for full gradient flow.
#[derive(Clone)]
pub struct MultiHeadAttention<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const H: usize = 8,
    M: AttentionMask = coeus_autograd::NullMask,
> {
    /// Query projection weight: `[d_model, d_model]`.
    pub w_q: Var<T, B>,
    /// Query projection bias: `[d_model]`.
    pub b_q: Option<Var<T, B>>,
    /// Key projection weight: `[d_model, d_model]`.
    pub w_k: Var<T, B>,
    /// Key projection bias: `[d_model]`.
    pub b_k: Option<Var<T, B>>,
    /// Value projection weight: `[d_model, d_model]`.
    pub w_v: Var<T, B>,
    /// Value projection bias: `[d_model]`.
    pub b_v: Option<Var<T, B>>,
    /// Output projection weight: `[d_model, d_model]`.
    pub w_o: Var<T, B>,
    /// Output projection bias: `[d_model]`.
    pub b_o: Option<Var<T, B>>,
    /// Model embedding dimension (must be divisible by `H`).
    pub d_model: usize,
    _mask: PhantomData<M>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const H: usize, M: AttentionMask>
    MultiHeadAttention<T, B, H, M>
{
    /// Construct a MHA layer with Kaiming-uniform initialized projection weights.
    ///
    /// # Panics
    /// Panics if `d_model % H != 0`.
    pub fn new(d_model: usize, bias: bool) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        assert!(
            H > 0 && d_model.is_multiple_of(H),
            "MultiHeadAttention: d_model ({d_model}) must be divisible by H ({H})"
        );
        let make_weight = || -> Var<T, B> {
            let mut v = Var::new(
                coeus_tensor::Tensor::zeros_on([d_model, d_model], &B::default()),
                true,
            );
            kaiming_uniform(&mut v, d_model);
            v
        };
        let make_bias = || -> Var<T, B> {
            Var::new(
                coeus_tensor::Tensor::zeros_on([d_model], &B::default()),
                true,
            )
        };
        Self {
            w_q: make_weight(),
            b_q: if bias { Some(make_bias()) } else { None },
            w_k: make_weight(),
            b_k: if bias { Some(make_bias()) } else { None },
            w_v: make_weight(),
            b_v: if bias { Some(make_bias()) } else { None },
            w_o: make_weight(),
            b_o: if bias { Some(make_bias()) } else { None },
            d_model,
            _mask: PhantomData,
        }
    }

    /// Cross-attention forward.
    ///
    /// - `query`: `[batch, seq_q, d_model]`
    /// - `key`:   `[batch, seq_k, d_model]`
    /// - `value`: `[batch, seq_k, d_model]`
    ///
    /// Returns `[batch, seq_q, d_model]`.
    pub fn forward_cross(
        &self,
        query: &Var<T, B>,
        key: &Var<T, B>,
        value: &Var<T, B>,
        key_padding_mask: Option<&Var<T, B>>,
    ) -> Var<T, B> {
        let d_head = self.d_model / H;
        let scale = T::one() / <T as Scalar>::from_f64((d_head as f64).sqrt());

        let q_shape = query.tensor.shape_cloned();
        let batch = q_shape[0];
        let seq_q = q_shape[1];

        let k_shape = key.tensor.shape_cloned();
        let seq_k = k_shape[1];

        // ── Project: [batch, seq, d_model] → [batch*seq, d_model] → matmul W → [batch*seq, d_model] → [batch, seq, d_model] ──
        let q_proj = self.project_3d(query, &self.w_q, &self.b_q, batch, seq_q);
        let k_proj = self.project_3d(key, &self.w_k, &self.b_k, batch, seq_k);
        let v_proj = self.project_3d(value, &self.w_v, &self.b_v, batch, seq_k);

        // ── Reshape to [batch, seq, H, d_head] ──
        let q_split = coeus_autograd::reshape(&q_proj, [batch, seq_q, H, d_head]);
        let k_split = coeus_autograd::reshape(&k_proj, [batch, seq_k, H, d_head]);
        let v_split = coeus_autograd::reshape(&v_proj, [batch, seq_k, H, d_head]);

        // ── Permute to [batch, H, seq, d_head] ──
        let q_perm = coeus_autograd::permute(&q_split, &[0, 2, 1, 3]);
        let k_perm = coeus_autograd::permute(&k_split, &[0, 2, 1, 3]);
        let v_perm = coeus_autograd::permute(&v_split, &[0, 2, 1, 3]);

        // ── Reshape to [batch * H, seq, d_head] ──
        let q_heads = coeus_autograd::reshape(&q_perm, [batch * H, seq_q, d_head]);
        let k_heads = coeus_autograd::reshape(&k_perm, [batch * H, seq_k, d_head]);
        let v_heads = coeus_autograd::reshape(&v_perm, [batch * H, seq_k, d_head]);

        // ── Scaled dot-product attention ──
        let (attn_out, _aw) = coeus_autograd::sdp_attention::<T, B, M>(
            &q_heads,
            &k_heads,
            &v_heads,
            key_padding_mask,
            scale,
        );

        // ── Reshape to [batch, H, seq_q, d_head] ──
        let merged_split = coeus_autograd::reshape(&attn_out, [batch, H, seq_q, d_head]);

        // ── Permute back to [batch, seq_q, H, d_head] ──
        let merged_perm = coeus_autograd::permute(&merged_split, &[0, 2, 1, 3]);

        // ── Merge heads: reshape to [batch, seq_q, self.d_model] ──
        let merged = coeus_autograd::reshape(&merged_perm, [batch, seq_q, self.d_model]);

        // ── Output projection ──
        self.project_3d(&merged, &self.w_o, &self.b_o, batch, seq_q)
    }

    /// Project a 3D `[batch, seq, d_model]` var via a `[d_model, d_model]` weight:
    /// - Reshape to `[batch*seq, d_model]` (tracked)
    /// - Matmul with `w^T` (tracked)
    /// - Reshape back to `[batch, seq, d_model]` (tracked)
    fn project_3d(
        &self,
        x: &Var<T, B>,
        w: &Var<T, B>,
        bias: &Option<Var<T, B>>,
        batch: usize,
        seq: usize,
    ) -> Var<T, B> {
        // Flatten [batch, seq, d_model] → [batch*seq, d_model] via tracked reshape
        let flat = coeus_autograd::reshape(x, [batch * seq, self.d_model]);
        // Project: matmul with W^T: [batch*seq, d_model] x [d_model, d_model]^T → [batch*seq, d_model]
        let w_t = coeus_autograd::transpose_2d(w);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        let out_flat = if let Some(ref b) = bias {
            coeus_autograd::add(&out_flat, b)
        } else {
            out_flat
        };
        // Unflatten [batch*seq, d_model] → [batch, seq, d_model] via tracked reshape
        coeus_autograd::reshape(&out_flat, [batch, seq, self.d_model])
    }
}

/// Self-attention `Module` impl (Q = K = V = input).
impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const H: usize, M: AttentionMask> Module<T, B>
    for MultiHeadAttention<T, B, H, M>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = vec![
            self.w_q.clone(),
            self.w_k.clone(),
            self.w_v.clone(),
            self.w_o.clone(),
        ];
        if let Some(ref b) = self.b_q {
            p.push(b.clone());
        }
        if let Some(ref b) = self.b_k {
            p.push(b.clone());
        }
        if let Some(ref b) = self.b_v {
            p.push(b.clone());
        }
        if let Some(ref b) = self.b_o {
            p.push(b.clone());
        }
        p
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        self.forward_cross(input, input, input, None)
    }
}
