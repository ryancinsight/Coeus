use leto::{LetoError, Result};

/// Largest dynamic rank the const-rank dispatch resolves. Coeus activations and
/// Apollo transforms stay well within this bound; ranks beyond it are a logged
/// error rather than silent truncation.
///
/// # Examples
///
/// The bound is a fixed constant, and dispatch rejects a rank that exceeds it:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::{elementwise_add_into, MAX_DISPATCH_RANK};
///
/// assert_eq!(MAX_DISPATCH_RANK, 6);
///
/// // A rank-7 tensor is beyond the dispatch bound.
/// let a = vec![0.0_f64; 128];
/// let la = Layout::new([2, 2, 2, 2, 2, 2, 2].into());
/// let mut out = vec![0.0_f64; 128];
/// assert!(elementwise_add_into(&la, &a, &la, &a, &la, &mut out).is_err());
/// ```
pub const MAX_DISPATCH_RANK: usize = 6;

/// Convert a dynamic-rank slice to a const-rank array for leto dispatch calls.
pub(crate) fn shape_n<const N: usize>(shape: &[usize]) -> Result<[usize; N]> {
    shape.try_into().map_err(
        |_: std::array::TryFromSliceError| LetoError::ShapeMismatch {
            lhs: vec![N],
            rhs: vec![shape.len()],
        },
    )
}

/// Provider-owned scaled dot-product attention dispatch.
pub mod attention;
/// Provider-owned regular and transposed convolution dispatch.
pub mod convolution;
/// Elementwise binary and unary operation dispatch (add, sub, mul, div, map).
pub mod elementwise;
/// Tensor initialization dispatch (from_shape_fn, uniform, normal).
pub mod init;
/// Layout metadata dispatch (reshape, permute, broadcast, contiguous).
pub mod layout;
/// Linear algebra dispatch (matmul, batched_matmul, accumulate variants).
pub mod linalg;
/// Reduction and scan dispatch (sum, mean, max, min, cumulative sum/product,
/// argmax, argmin).
pub mod reductions;
/// Rotary half-vector dispatch.
pub mod rotary;
/// Sparse matrix dispatch (CSR mat-vec and mat-mat).
pub mod sparse;
/// Scalar-preserving stateful parameter-update dispatch.
pub mod stateful_update;
/// Structural tensor ops dispatch (pad, concat, split, stack).
pub mod structural;

pub use attention::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionBackward, AttentionForward, AttentionGradientTargets, AttentionScalar,
};
pub use convolution::{
    convolution_backward_accumulate, convolution_forward_into,
    convolution_transposed_backward_accumulate, convolution_transposed_forward_into,
    ConvolutionBackward, ConvolutionForward, ConvolutionGradients, ReadOperand, WriteOperand,
};
pub use elementwise::{
    elementwise_add_into, elementwise_binary_assign, elementwise_binary_into,
    elementwise_unary_into,
};
pub use init::{
    from_shape_fn_values, normal_values, normal_values_into, uniform_values, uniform_values_into,
};
pub use layout::{
    broadcast_layout, broadcast_shape, contiguous_values, permute_layout, reshape_layout,
};
pub use linalg::{
    batched_matmul_accumulate_into, batched_matmul_into, matmul_accumulate_into, matmul_into,
};
pub use reductions::{
    argmax_into, argmin_into, cumprod_into, cumsum_into, reduce_into, suffix_prod_into,
    suffix_sum_into,
};
pub use rotary::{prepare_rotate_half_input, rotate_half_into, RotateHalfPlan};
pub use sparse::{spmm_into, spmv_into, CsrDispatch};
pub use stateful_update::{
    stateful_update, validate_stateful_update, StatefulUpdateDispatchRule, StatefulUpdateOperands,
    StatefulUpdateState, StatefulUpdateValidation, StatefulUpdateValidationState,
    MAX_STATEFUL_UPDATE_RANK,
};
pub use structural::{concat_values, pad_values, split_values, stack_values};
