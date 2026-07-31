//! Zero-copy Coeus-to-Leto scaled dot-product attention dispatch.

use core::num::NonZeroUsize;

use coeus_core::Float;
use leto::{ArrayView, ArrayViewMut, LetoError};
use leto_ops::{
    AttentionError, AttentionGradients as LetoAttentionGradients, AttentionMask, AttentionOperand,
    AttentionResult, GroupedKeepMask, RealScalar,
};

use super::convolution::{ReadOperand, WriteOperand};
use crate::{to_leto_view, to_leto_view_mut};

/// Scalars supported by Leto's scaled dot-product attention provider.
///
/// Leto currently defines its real-field attention contract for `f32` and
/// `f64`. This trait keeps that provider support boundary explicit at Coeus's
/// generic dispatch seam.
pub trait AttentionScalar: Float + RealScalar {
    #[doc(hidden)]
    fn attention_forward(
        query: &ArrayView<'_, Self, 3>,
        key: &ArrayView<'_, Self, 3>,
        value: &ArrayView<'_, Self, 3>,
        mask: AttentionMask<'_, Self>,
        scale: Self,
        output: &mut ArrayViewMut<'_, Self, 3>,
        weights: &mut ArrayViewMut<'_, Self, 3>,
    ) -> AttentionResult<()>;

    #[doc(hidden)]
    fn attention_backward(
        output_gradient: &ArrayView<'_, Self, 3>,
        query: &ArrayView<'_, Self, 3>,
        key: &ArrayView<'_, Self, 3>,
        value: &ArrayView<'_, Self, 3>,
        weights: &ArrayView<'_, Self, 3>,
        scale: Self,
        gradients: LetoAttentionGradients<'_, Self>,
    ) -> AttentionResult<()>;
}

macro_rules! impl_attention_scalar {
    ($scalar:ty) => {
        impl AttentionScalar for $scalar {
            #[inline]
            fn attention_forward(
                query: &ArrayView<'_, Self, 3>,
                key: &ArrayView<'_, Self, 3>,
                value: &ArrayView<'_, Self, 3>,
                mask: AttentionMask<'_, Self>,
                scale: Self,
                output: &mut ArrayViewMut<'_, Self, 3>,
                weights: &mut ArrayViewMut<'_, Self, 3>,
            ) -> AttentionResult<()> {
                leto_ops::scaled_dot_product_attention_into(
                    query, key, value, mask, scale, output, weights,
                )
            }

            #[inline]
            fn attention_backward(
                output_gradient: &ArrayView<'_, Self, 3>,
                query: &ArrayView<'_, Self, 3>,
                key: &ArrayView<'_, Self, 3>,
                value: &ArrayView<'_, Self, 3>,
                weights: &ArrayView<'_, Self, 3>,
                scale: Self,
                gradients: LetoAttentionGradients<'_, Self>,
            ) -> AttentionResult<()> {
                leto_ops::scaled_dot_product_attention_backward_accumulate(
                    output_gradient,
                    query,
                    key,
                    value,
                    weights,
                    scale,
                    gradients,
                )
            }
        }
    };
}

impl_attention_scalar!(f32);
impl_attention_scalar!(f64);

/// Borrowed operands for scaled dot-product attention forward.
pub struct AttentionForward<'a, T> {
    /// Query values in `[batch, query, feature]` order.
    pub query: ReadOperand<'a, T>,
    /// Key values in `[batch, key, feature]` order.
    pub key: ReadOperand<'a, T>,
    /// Value values in `[batch, key, value]` order.
    pub value: ReadOperand<'a, T>,
    /// Optional keep mask. Rank two means `[group, key]`; rank one means `[key]`.
    pub keep_mask: Option<ReadOperand<'a, T>>,
    /// Whether query `i` may attend only to keys `j <= i`.
    pub is_causal: bool,
    /// Score multiplier applied before softmax.
    pub scale: T,
    /// Caller-owned result in `[batch, query, value]` order.
    pub output: WriteOperand<'a, T>,
    /// Caller-owned post-softmax weights in `[batch, query, key]` order.
    pub weights: WriteOperand<'a, T>,
}

/// Selected additive scaled dot-product attention gradient targets.
pub struct AttentionGradientTargets<'a, T> {
    /// Optional query-gradient destination.
    pub query: Option<WriteOperand<'a, T>>,
    /// Optional key-gradient destination.
    pub key: Option<WriteOperand<'a, T>>,
    /// Optional value-gradient destination.
    pub value: Option<WriteOperand<'a, T>>,
}

/// Borrowed operands for scaled dot-product attention backward.
pub struct AttentionBackward<'a, T> {
    /// Gradient of the forward output.
    pub output_gradient: ReadOperand<'a, T>,
    /// Query values from the forward pass.
    pub query: ReadOperand<'a, T>,
    /// Key values from the forward pass.
    pub key: ReadOperand<'a, T>,
    /// Value values from the forward pass.
    pub value: ReadOperand<'a, T>,
    /// Post-softmax weights from the forward pass.
    pub weights: ReadOperand<'a, T>,
    /// Score multiplier used by the forward pass.
    pub scale: T,
    /// Selected additive gradient destinations.
    pub gradients: AttentionGradientTargets<'a, T>,
}

fn map_layout_error(operand: AttentionOperand, source: LetoError) -> AttentionError {
    AttentionError::Layout { operand, source }
}

fn read_view<'a, T>(
    operand: AttentionOperand,
    value: ReadOperand<'a, T>,
) -> AttentionResult<ArrayView<'a, T, 3>> {
    to_leto_view::<T, 3>(value.layout, value.data)
        .map_err(|source| map_layout_error(operand, source))
}

fn write_view<'a, T>(
    operand: AttentionOperand,
    value: WriteOperand<'a, T>,
) -> AttentionResult<ArrayViewMut<'a, T, 3>> {
    to_leto_view_mut::<T, 3>(value.layout, value.data)
        .map_err(|source| map_layout_error(operand, source))
}

fn mask_policy<'a, T>(mask: Option<ArrayView<'a, T, 3>>, is_causal: bool) -> AttentionMask<'a, T> {
    match (mask, is_causal) {
        (Some(mask), true) => AttentionMask::CausalKeep(mask),
        (Some(mask), false) => AttentionMask::Keep(mask),
        (None, true) => AttentionMask::Causal,
        (None, false) => AttentionMask::Unmasked,
    }
}

fn rank_two_mask_view<T>(mask: ReadOperand<'_, T>) -> AttentionResult<ArrayView<'_, T, 2>> {
    to_leto_view::<T, 2>(mask.layout, mask.data)
        .map_err(|source| map_layout_error(AttentionOperand::Mask, source))
}

fn dispatch_forward<T: AttentionScalar>(
    query: &ArrayView<'_, T, 3>,
    key: &ArrayView<'_, T, 3>,
    value: &ArrayView<'_, T, 3>,
    mask: AttentionMask<'_, T>,
    scale: T,
    output: &mut ArrayViewMut<'_, T, 3>,
    weights: &mut ArrayViewMut<'_, T, 3>,
) -> AttentionResult<()> {
    T::attention_forward(query, key, value, mask, scale, output, weights)
}

/// Execute scaled dot-product attention directly through Leto over borrowed
/// Coeus storage.
///
/// Rank-three tensor strides and offsets are preserved. A rank-two keep mask
/// uses `[group, key]` semantics and is borrowed directly by Leto's grouped
/// policy. Backend selection and validation happen once per complete operation.
///
/// # Errors
///
/// Returns [`leto_ops::AttentionError`] when an operand layout, shape, mask,
/// finite-value, or arithmetic contract is invalid.
pub fn scaled_dot_product_attention_into<T: AttentionScalar>(
    operands: AttentionForward<'_, T>,
) -> AttentionResult<()> {
    let mask = match operands.keep_mask {
        Some(mask) if mask.layout.ndim() == 2 => mask,
        _ => {
            let query = read_view(AttentionOperand::Query, operands.query)?;
            let key = read_view(AttentionOperand::Key, operands.key)?;
            let value = read_view(AttentionOperand::Value, operands.value)?;
            let mask = operands
                .keep_mask
                .map(|mask| read_view(AttentionOperand::Mask, mask))
                .transpose()?;
            let mut output = write_view(AttentionOperand::Output, operands.output)?;
            let mut weights = write_view(AttentionOperand::Weights, operands.weights)?;
            return dispatch_forward(
                &query,
                &key,
                &value,
                mask_policy(mask, operands.is_causal),
                operands.scale,
                &mut output,
                &mut weights,
            );
        }
    };

    let query = read_view(AttentionOperand::Query, operands.query)?;
    let key = read_view(AttentionOperand::Key, operands.key)?;
    let value = read_view(AttentionOperand::Value, operands.value)?;
    let mask = rank_two_mask_view(mask)?;
    let batch = query.shape()[0];
    let groups = mask.shape()[0];
    let target = [batch, query.shape()[1], mask.shape()[1]];
    if groups == 0 || batch % groups != 0 {
        return Err(AttentionError::MaskShape {
            actual: [groups, 1, mask.shape()[1]],
            target,
        });
    }
    let Some(batches_per_group) = NonZeroUsize::new(batch / groups) else {
        return Err(AttentionError::MaskShape {
            actual: [groups, 1, mask.shape()[1]],
            target,
        });
    };
    let grouped = GroupedKeepMask::new(mask, batches_per_group);
    let policy = if operands.is_causal {
        AttentionMask::CausalGroupedKeep(grouped)
    } else {
        AttentionMask::GroupedKeep(grouped)
    };
    let mut output = write_view(AttentionOperand::Output, operands.output)?;
    let mut weights = write_view(AttentionOperand::Weights, operands.weights)?;
    dispatch_forward(
        &query,
        &key,
        &value,
        policy,
        operands.scale,
        &mut output,
        &mut weights,
    )
}

/// Accumulate selected scaled dot-product attention gradients directly through
/// Leto over borrowed Coeus storage.
///
/// # Errors
///
/// Returns [`leto_ops::AttentionError`] when an operand layout, shape,
/// finite-value, probability-weight, or additive-gradient contract is invalid.
pub fn scaled_dot_product_attention_backward_accumulate<T: AttentionScalar>(
    operands: AttentionBackward<'_, T>,
) -> AttentionResult<()> {
    let output_gradient = read_view(AttentionOperand::OutputGradient, operands.output_gradient)?;
    let query = read_view(AttentionOperand::Query, operands.query)?;
    let key = read_view(AttentionOperand::Key, operands.key)?;
    let value = read_view(AttentionOperand::Value, operands.value)?;
    let weights = read_view(AttentionOperand::Weights, operands.weights)?;
    let query_gradient = operands
        .gradients
        .query
        .map(|value| write_view(AttentionOperand::QueryGradient, value))
        .transpose()?;
    let key_gradient = operands
        .gradients
        .key
        .map(|value| write_view(AttentionOperand::KeyGradient, value))
        .transpose()?;
    let value_gradient = operands
        .gradients
        .value
        .map(|value| write_view(AttentionOperand::ValueGradient, value))
        .transpose()?;

    T::attention_backward(
        &output_gradient,
        &query,
        &key,
        &value,
        &weights,
        operands.scale,
        LetoAttentionGradients::new(query_gradient, key_gradient, value_gradient),
    )
}
