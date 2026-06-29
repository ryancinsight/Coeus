//! Reverse-mode automatic differentiation engine built on the Coeus tensor and ops stacks.
//!
//! # Key types
//! - [`Var<T, B>`](var::Var) — a tracked tensor carrying an optional gradient accumulator and
//!   an optional `Arc<dyn BackwardNode<T, B>>` creator link.
//! - [`BackwardNode`] — trait implemented by per-op nodes; each node stores
//!   saved tensors and accumulates gradients into its inputs during the reverse pass.
//! - [`Var::backward`](var::Var::backward) — triggers topological traversal of the computation
//!   graph seeded with a ones tensor, propagating gradients to all `requires_grad` leaves.
//!
//! All differentiable ops in [`ops`] are thin wrappers that call [`ops::arithmetic::binary_op`] or
//! [`ops::activation::unary_op`], which construct the forward result and attach the creator node.

// ── Coeus Autograd ──
// Automatic differentiation engine with computational graph.
#![allow(
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::get_first
)]
#![deny(missing_docs)]

/// Backward-pass graph traversal and gradient propagation.
pub mod backward;
pub(crate) mod grad_buffer;
/// Thread-local autograd recording mode (no-grad scopes).
pub mod grad_mode;
/// Computation graph node trait and implementations.
pub mod node;
/// Differentiable operations that build the autograd graph.
pub mod ops;
/// The differentiable variable type.
pub mod var;

pub use grad_buffer::GradBuffer;
pub use grad_mode::{
    is_grad_enabled, is_no_grad_enabled, no_grad_guard, pop_no_grad, push_no_grad, NoGradGuard,
};
pub use node::BackwardNode;
pub use ops::{
    abs,
    add,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    batchnorm1d,
    batchnorm2d,
    batchnorm3d,
    bce_with_logits,
    binary_cross_entropy,
    // Shape ops
    broadcast_to,
    cat,
    ceil,
    celu,
    clamp,
    contiguous,
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    // Trigonometric
    cos,
    cosine_embedding_loss,
    cross_entropy_loss,
    cumprod,
    cumsum,
    // Diagonal ops
    diag,
    diagonal,
    div,
    dropout,
    // Index ops
    einsum,
    einsum3,
    elu,
    embedding,
    embedding_with_padding_idx,
    exp,
    flip,
    floor,
    gather,
    gelu,
    gelu_tanh,
    hardshrink,
    hardsigmoid,
    hardswish,
    hardtanh,
    huber_loss,
    index_select,
    kl_divergence,
    l1_loss,
    layernorm,
    leaky_relu,
    log,
    log_softmax,
    log_sum_exp,
    margin_ranking_loss,
    masked_fill,
    matmul,
    // New axis reductions
    max_axis,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    mean,
    mean_axis,
    min_axis,
    mish,
    mul,
    // New unary math ops
    neg,
    multi_margin,
    nll_loss,
    pairwise_distance,
    poisson_nll,
    soft_margin,
    norm,
    norm_p,
    norm_p_axis,
    pack_pairs,
    pad,
    permute,
    pow,
    prelu,
    recip,
    relu,
    reshape,
    rmsnorm,
    roll,
    round,
    scalar_add,
    scalar_div,
    // New scalar arithmetic
    scalar_mul,
    scalar_sub,
    sdp_attention,
    sigmoid,
    sign,
    silu,
    // Trigonometric
    sin,
    slice,
    softmax,
    softplus,
    softshrink,
    softsign,
    sparse_matmul,
    sparse_matmul_coo,
    split,
    sqrt,
    squeeze,
    stack,
    sub,
    sum,
    sum_axis,
    tanh,
    threshold,
    // Tile / repeat
    tile,
    transpose,
    transpose_2d,
    // Triangular masking
    tril,
    triu,
    trunc,
    unsqueeze,
    // Shape ops
    where_cond,
    AttentionMask,
    BatchNormArgs,
    CausalMask,
    NullMask,
    VarScalarExt,
};

pub use var::Var;
