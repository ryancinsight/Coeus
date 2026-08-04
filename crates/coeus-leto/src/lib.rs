#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! # coeus-leto
//!
//! The const-rank dispatch shim that lets coeus delegate CPU array kernels to
//! [`leto`], per leto ADR 0002 (`docs/adr/0002-coeus-rank-boundary.md`).
//!
//! coeus carries a **dynamic-rank** [`coeus_core::Layout`] (rank held at
//! runtime in a `SmallVec`), while leto is **const-rank** (`Layout<const N>`),
//! which is the source of its compile-time shape safety and monomorphized,
//! allocation-free traversal. Rather than fork leto into a dynamic-rank model,
//! this crate resolves coeus's runtime rank to a leto `const N` through a
//! bounded `match` ([`dispatch`]) and calls the monomorphized leto kernel. The
//! shim lives here, in the consumer, so leto stays purely const-rank.
//!
//! This is the consolidation seam: coeus's CPU array operations route through
//! one authoritative leto kernel set instead of a duplicated traversal layer.

/// Zero-copy conversion from coeus dynamic-rank layouts to leto const-rank views.
pub mod convert;
/// Dynamic-rank to const-rank operation dispatch.
pub mod dispatch;

pub use convert::{to_leto_layout, to_leto_view, to_leto_view_mut};
pub use dispatch::{
    argmax_into, argmin_into, batched_matmul_accumulate_into, batched_matmul_into,
    broadcast_layout, broadcast_shape, concat_values, contiguous_values,
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into, cumprod_into,
    cumsum_into, elementwise_add_into, elementwise_binary_assign, elementwise_binary_into,
    elementwise_unary_into, from_shape_fn_values, matmul_accumulate_into, matmul_into,
    normal_values, normal_values_into, pad_values, permute_layout, prepare_rotate_half_input,
    reduce_into, reshape_layout, rotate_half_into,
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    split_values, spmm_into, spmv_into, stack_values, stateful_update, suffix_prod_into,
    suffix_sum_into, uniform_values, uniform_values_into, validate_stateful_update,
    AttentionBackward, AttentionForward, AttentionGradientTargets, AttentionScalar,
    ConvolutionBackward, ConvolutionForward, ConvolutionGradients, CsrDispatch, ReadOperand,
    RotateHalfPlan, StatefulUpdateDispatchRule, StatefulUpdateOperands, StatefulUpdateState,
    StatefulUpdateValidation, StatefulUpdateValidationState, WriteOperand, MAX_DISPATCH_RANK,
    MAX_STATEFUL_UPDATE_RANK,
};
pub use leto_ops::RealScalar as RandomScalar;
