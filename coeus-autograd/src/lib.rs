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

pub mod backward;
pub(crate) mod grad_buffer;
pub mod node;
pub mod ops;
pub mod var;

pub use grad_buffer::GradBuffer;
pub use node::BackwardNode;
pub use ops::{
    abs,
    add,
    avg_pool2d,
    avg_pool3d,
    batchnorm1d,
    batchnorm2d,
    batchnorm3d,
    binary_cross_entropy,
    // Shape ops
    broadcast_to,
    cat,
    ceil,
    clamp,
    contiguous,
    conv1d,
    conv2d,
    conv3d,
    // Trigonometric
    cos,
    cosine_embedding_loss,
    cross_entropy_loss,
    cumsum,
    div,
    dropout,
    // Index ops
    einsum,
    elu,
    embedding,
    exp,
    flip,
    floor,
    gather,
    gelu,
    gelu_tanh,
    huber_loss,
    index_select,
    layernorm,
    leaky_relu,
    log,
    log_softmax,
    log_sum_exp,
    masked_fill,
    matmul,
    // New axis reductions
    max_axis,
    max_pool2d,
    max_pool3d,
    mean,
    mean_axis,
    min_axis,
    mish,
    mul,
    // New unary math ops
    neg,
    nll_loss,
    norm,
    norm_p,
    norm_p_axis,
    pad,
    permute,
    pow,
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
    sparse_matmul,
    split,
    sqrt,
    squeeze,
    stack,
    sub,
    sum,
    sum_axis,
    tanh,
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
    BatchNorm1dArgs,
    BatchNorm2dArgs,
    BatchNorm3dArgs,
    CausalMask,
    NullMask,
};

pub use var::Var;
