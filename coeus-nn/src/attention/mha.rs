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

/// Borrowed projection parameters for functional multi-head attention.
pub struct MhaProjectionParams<'a, T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Query projection weight `[d_model, d_model]`.
    pub w_q: &'a Var<T, B>,
    /// Optional query projection bias `[d_model]`.
    pub b_q: Option<&'a Var<T, B>>,
    /// Key projection weight `[d_model, d_model]`.
    pub w_k: &'a Var<T, B>,
    /// Optional key projection bias `[d_model]`.
    pub b_k: Option<&'a Var<T, B>>,
    /// Value projection weight `[d_model, d_model]`.
    pub w_v: &'a Var<T, B>,
    /// Optional value projection bias `[d_model]`.
    pub b_v: Option<&'a Var<T, B>>,
    /// Output projection weight `[d_model, d_model]`.
    pub w_o: &'a Var<T, B>,
    /// Optional output projection bias `[d_model]`.
    pub b_o: Option<&'a Var<T, B>>,
}

/// Functional cross-attention helper shared by module and bindings.
///
/// Applies projections and scaled dot-product attention with explicit weights/biases
/// and returns output shape `[batch, seq_q, d_model]`.
///
/// # Panics
///
/// Panics when `w_q` is not square or when `d_model` is not divisible by `H`.
pub fn multi_head_attention_cross<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    const H: usize,
    M: AttentionMask,
>(
    query: &Var<T, B>,
    key: &Var<T, B>,
    value: &Var<T, B>,
    params: MhaProjectionParams<'_, T, B>,
    key_padding_mask: Option<&Var<T, B>>,
) -> Var<T, B> {
    let wq_shape = params.w_q.tensor.shape_cloned();
    assert!(
        wq_shape.len() == 2 && wq_shape[0] == wq_shape[1],
        "MultiHeadAttention: w_q must be square [d_model, d_model], got shape {:?}",
        wq_shape
    );
    let d_model = wq_shape[0];
    assert!(
        H > 0 && d_model.is_multiple_of(H),
        "MultiHeadAttention: d_model ({d_model}) must be divisible by H ({H})"
    );

    let d_head = d_model / H;
    let scale = T::one() / <T as Scalar>::from_f64((d_head as f64).sqrt());

    let q_shape = query.tensor.shape_cloned();
    let batch = q_shape[0];
    let seq_q = q_shape[1];

    let k_shape = key.tensor.shape_cloned();
    let seq_k = k_shape[1];

    let q_proj = project_3d(query, params.w_q, params.b_q, batch, seq_q, d_model);
    let k_proj = project_3d(key, params.w_k, params.b_k, batch, seq_k, d_model);
    let v_proj = project_3d(value, params.w_v, params.b_v, batch, seq_k, d_model);

    let q_split = coeus_autograd::reshape(&q_proj, [batch, seq_q, H, d_head]);
    let k_split = coeus_autograd::reshape(&k_proj, [batch, seq_k, H, d_head]);
    let v_split = coeus_autograd::reshape(&v_proj, [batch, seq_k, H, d_head]);

    let q_perm = coeus_autograd::permute(&q_split, &[0, 2, 1, 3]);
    let k_perm = coeus_autograd::permute(&k_split, &[0, 2, 1, 3]);
    let v_perm = coeus_autograd::permute(&v_split, &[0, 2, 1, 3]);

    let q_heads = coeus_autograd::reshape(&q_perm, [batch * H, seq_q, d_head]);
    let k_heads = coeus_autograd::reshape(&k_perm, [batch * H, seq_k, d_head]);
    let v_heads = coeus_autograd::reshape(&v_perm, [batch * H, seq_k, d_head]);

    let (attn_out, _aw) = coeus_autograd::sdp_attention::<T, B, M>(
        &q_heads,
        &k_heads,
        &v_heads,
        key_padding_mask,
        scale,
    );

    let merged_split = coeus_autograd::reshape(&attn_out, [batch, H, seq_q, d_head]);
    let merged_perm = coeus_autograd::permute(&merged_split, &[0, 2, 1, 3]);
    let merged = coeus_autograd::reshape(&merged_perm, [batch, seq_q, d_model]);

    project_3d(&merged, params.w_o, params.b_o, batch, seq_q, d_model)
}

/// Project a 3D `[batch, seq, d_model]` var via a `[d_model, d_model]` weight:
/// - Reshape to `[batch*seq, d_model]` (tracked)
/// - Matmul with `w^T` (tracked)
/// - Reshape back to `[batch, seq, d_model]` (tracked)
fn project_3d<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    w: &Var<T, B>,
    bias: Option<&Var<T, B>>,
    batch: usize,
    seq: usize,
    d_model: usize,
) -> Var<T, B> {
    let flat = coeus_autograd::reshape(x, [batch * seq, d_model]);
    let w_t = coeus_autograd::transpose_2d(w);
    let out_flat = coeus_autograd::matmul(&flat, &w_t);
    let out_flat = if let Some(b) = bias {
        coeus_autograd::add(&out_flat, b)
    } else {
        out_flat
    };
    coeus_autograd::reshape(&out_flat, [batch, seq, d_model])
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
        multi_head_attention_cross::<T, B, H, M>(
            query,
            key,
            value,
            MhaProjectionParams {
                w_q: &self.w_q,
                b_q: self.b_q.as_ref(),
                w_k: &self.w_k,
                b_k: self.b_k.as_ref(),
                w_v: &self.w_v,
                b_v: self.b_v.as_ref(),
                w_o: &self.w_o,
                b_o: self.b_o.as_ref(),
            },
            key_padding_mask,
        )
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
