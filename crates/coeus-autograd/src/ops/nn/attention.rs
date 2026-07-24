// ── Autograd node: Scaled Dot-Product Attention ──
//
// Monomorphized struct-based BackwardNode following the Conv/Pool pattern.
// `M: AttentionMask` is a ZST; the causal branch is selected at compile time
// by DCE on `M::IS_CAUSAL`.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{Float, Scalar};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST marker trait for attention mask strategies.
///
/// Implementations carry only a compile-time `IS_CAUSAL` constant.
/// DCE eliminates the inactive branch in `forward`; no runtime branching occurs.
pub trait AttentionMask: 'static + Copy + Clone + Send + Sync {
    /// `true` iff the mask is a lower-triangular causal mask.
    const IS_CAUSAL: bool;
}

/// Causal (lower-triangular) attention mask — positions j > i are masked to −∞.
#[derive(Debug, Clone, Copy, Default)]
pub struct CausalMask;
impl AttentionMask for CausalMask {
    const IS_CAUSAL: bool = true;
}

/// No masking (full attention).
#[derive(Debug, Clone, Copy, Default)]
pub struct NullMask;
impl AttentionMask for NullMask {
    const IS_CAUSAL: bool = false;
}

// ── Backward Node ──────────────────────────────────────────────────────────

/// Autograd node for scaled dot-product attention.
///
/// Stores the forward inputs and the post-softmax attention weights needed for
/// computing gradients w.r.t. Q, K, and V.
pub struct ScaledDotProductAttnNode<
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    M: AttentionMask,
> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// [Q_var, K_var, V_var]
    pub inputs: Vec<Var<T, B>>,
    /// Saved query tensor for backward.
    pub q_clone: Tensor<T, B>,
    /// Saved key tensor for backward.
    pub k_clone: Tensor<T, B>,
    /// Saved value tensor for backward.
    pub v_clone: Tensor<T, B>,
    /// Post-softmax attention weight matrix `[batch, seq_q, seq_k]`.
    pub attn_weights: Tensor<T, B>,
    /// Scaling factor `1/sqrt(head_dim)`.
    pub scale: T,
    /// Zero-sized phantom to bind the mask type parameter.
    pub _mask: std::marker::PhantomData<M>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, M: AttentionMask> BackwardNode<T, B>
    for ScaledDotProductAttnNode<T, B, M>
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "sdp_attention"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();

        let need_gq = input_grads.first().and_then(|g| g.as_ref()).is_some();
        let need_gk = input_grads.get(1).and_then(|g| g.as_ref()).is_some();
        let need_gv = input_grads.get(2).and_then(|g| g.as_ref()).is_some();

        if !need_gq && !need_gk && !need_gv {
            return;
        }

        let mut grad_q = need_gq.then(|| Tensor::zeros_on(self.q_clone.shape_cloned(), &backend));
        let mut grad_k = need_gk.then(|| Tensor::zeros_on(self.k_clone.shape_cloned(), &backend));
        let mut grad_v = need_gv.then(|| Tensor::zeros_on(self.v_clone.shape_cloned(), &backend));

        // All six borrow paths handled in coeus_ops::scaled_dot_product_attention_backward.
        coeus_ops::scaled_dot_product_attention_backward(
            grad_out,
            &self.q_clone,
            &self.k_clone,
            &self.v_clone,
            &self.attn_weights,
            self.scale,
            grad_q.as_mut(),
            grad_k.as_mut(),
            grad_v.as_mut(),
            &backend,
        );

        if let (Some(acc), Some(gq)) = (input_grads.first().and_then(|g| g.as_ref()), grad_q) {
            let lock = acc.write();
            coeus_ops::add_assign(lock, &gq, &backend);
        }
        if let (Some(acc), Some(gk)) = (input_grads.get(1).and_then(|g| g.as_ref()), grad_k) {
            let lock = acc.write();
            coeus_ops::add_assign(lock, &gk, &backend);
        }
        if let (Some(acc), Some(gv)) = (input_grads.get(2).and_then(|g| g.as_ref()), grad_v) {
            let lock = acc.write();
            coeus_ops::add_assign(lock, &gv, &backend);
        }
    }
}

// ── Public tracked function ────────────────────────────────────────────────

/// Tracked scaled dot-product attention.
///
/// `M::IS_CAUSAL` selects the masking strategy; dead code is eliminated at
/// monomorphization time — no runtime branch.
///
/// Returns `(attn_output, attn_weights)`. `attn_weights` is detached (no
/// further gradient tracking needed for the basic use-case; MHA uses it
/// as an intermediate).
pub fn sdp_attention<T: Float, B: coeus_ops::BackendOps<T> + Default, M: AttentionMask>(
    query: &Var<T, B>,
    key: &Var<T, B>,
    value: &Var<T, B>,
    key_padding_mask: Option<&Var<T, B>>,
    scale: T,
) -> (Var<T, B>, Tensor<T, B>) {
    let backend = B::default();

    let (out_tensor, attn_weights) = coeus_ops::scaled_dot_product_attention(
        &query.tensor,
        &key.tensor,
        &value.tensor,
        key_padding_mask.map(|m| &m.tensor),
        M::IS_CAUSAL,
        scale,
        &backend,
    );

    let requires_grad = crate::grad_mode::should_track_var(query)
        || crate::grad_mode::should_track_var(key)
        || crate::grad_mode::should_track_var(value)
        || key_padding_mask.is_some_and(|m| crate::grad_mode::should_track_var(m));

    if !requires_grad {
        return (Var::new(out_tensor, false), attn_weights);
    }

    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(
        out_tensor.shape_cloned(),
        &backend,
    )));
    let grad = Some(output_grad.clone());

    let mut inputs = vec![query.clone(), key.clone(), value.clone()];
    if let Some(mask) = key_padding_mask {
        inputs.push(mask.clone());
    }

    let node = ScaledDotProductAttnNode::<T, B, M> {
        output_grad,
        inputs,
        q_clone: query.tensor.clone(),
        k_clone: key.tensor.clone(),
        v_clone: value.tensor.clone(),
        attn_weights: attn_weights.clone(),
        scale,
        _mask: std::marker::PhantomData,
    };
    let creator = Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>);

    let out_var = Var {
        tensor: out_tensor,
        grad,
        creator,
    };
    (out_var, attn_weights)
}
