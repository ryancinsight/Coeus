use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use coeus_hephaestus::AttentionProvider;
use hephaestus_core::{
    plan_attention_backward, plan_attention_forward, AttentionBackwardOperands, AttentionCausality,
    AttentionForwardOperands, AttentionOps as HephaestusAttentionOps, HephaestusError,
};
use leto::{ArrayView, ArrayViewMut};
use leto_ops::{
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
    AttentionGradients, AttentionMask, GroupedKeepMask,
};

use crate::device::*;

#[derive(Clone, Copy, Default)]
pub(crate) struct TestAttentionOps;

pub(crate) struct TestPreparedForward<'a> {
    operands: AttentionForwardOperands<'a, TestBuffer<f32>, f32>,
}

pub(crate) struct TestPreparedBackward<'a> {
    operands: AttentionBackwardOperands<'a, TestBuffer<f32>, f32>,
}

pub(crate) static FORWARD_BATCHES: AtomicUsize = AtomicUsize::new(0);
pub(crate) static FORWARD_MASK_GROUP: AtomicUsize = AtomicUsize::new(0);
pub(crate) static FORWARD_CAUSAL: AtomicBool = AtomicBool::new(false);
pub(crate) static BACKWARD_GRADIENTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static FAIL_FORWARD: AtomicBool = AtomicBool::new(false);

impl HephaestusAttentionOps<TestDevice, f32> for TestAttentionOps {
    type PreparedForward<'a> = TestPreparedForward<'a>;
    type PreparedBackward<'a> = TestPreparedBackward<'a>;

    fn prepare_attention_forward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionForwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedForward<'a>> {
        plan_attention_forward(&operands, false)?;
        FORWARD_BATCHES.store(operands.query.layout.shape()[0], Ordering::SeqCst);
        FORWARD_CAUSAL.store(
            operands.mask.causality() == AttentionCausality::Causal,
            Ordering::SeqCst,
        );
        FORWARD_MASK_GROUP.store(
            operands
                .mask
                .grouped_keep()
                .map_or(0, |mask| mask.heads_per_batch().get()),
            Ordering::SeqCst,
        );
        Ok(TestPreparedForward { operands })
    }

    fn dispatch_attention_forward(
        &self,
        _device: &TestDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> hephaestus_core::Result<()> {
        if FAIL_FORWARD.swap(false, Ordering::SeqCst) {
            return Err(HephaestusError::DispatchFailed {
                message: "injected provider failure".into(),
            });
        }
        let operands = &prepared.operands;
        let query_values = operands
            .query
            .buffer
            .values
            .lock()
            .expect("test query buffer lock");
        let key_values = operands
            .key
            .buffer
            .values
            .lock()
            .expect("test key buffer lock");
        let value_values = operands
            .value
            .buffer
            .values
            .lock()
            .expect("test value buffer lock");
        let query =
            ArrayView::try_new(*operands.query.layout, &query_values).map_err(attention_error)?;
        let key = ArrayView::try_new(*operands.key.layout, &key_values).map_err(attention_error)?;
        let value =
            ArrayView::try_new(*operands.value.layout, &value_values).map_err(attention_error)?;
        let keep = operands.mask.grouped_keep();
        let keep_values = keep.map(|keep| {
            keep.view()
                .buffer
                .values
                .lock()
                .expect("test attention mask lock")
        });
        let keep_view = keep_values
            .as_ref()
            .zip(keep)
            .map(|(values, keep)| {
                ArrayView::try_new(*keep.view().layout, values.as_slice()).map_err(attention_error)
            })
            .transpose()?;
        let grouped = keep_view.map(|view| {
            (
                view,
                keep.expect("invariant: keep view matches keep descriptor")
                    .heads_per_batch(),
            )
        });
        let mask = grouped.map_or(
            if operands.mask.causality() == AttentionCausality::Causal {
                AttentionMask::Causal
            } else {
                AttentionMask::Unmasked
            },
            |(view, heads_per_batch)| {
                let grouped = GroupedKeepMask::new(view, heads_per_batch);
                if operands.mask.causality() == AttentionCausality::Causal {
                    AttentionMask::CausalGroupedKeep(grouped)
                } else {
                    AttentionMask::GroupedKeep(grouped)
                }
            },
        );
        let mut output_values = operands
            .output
            .buffer
            .values
            .lock()
            .expect("test attention output lock");
        let mut weight_values = operands
            .weights
            .buffer
            .values
            .lock()
            .expect("test attention weights lock");
        let mut output = ArrayViewMut::try_new(*operands.output.layout, &mut output_values)
            .map_err(attention_error)?;
        let mut weights = ArrayViewMut::try_new(*operands.weights.layout, &mut weight_values)
            .map_err(attention_error)?;
        scaled_dot_product_attention_into(
            &query,
            &key,
            &value,
            mask,
            operands.scale,
            &mut output,
            &mut weights,
        )
        .map_err(attention_error)
    }

    fn prepare_attention_backward<'a>(
        &self,
        _device: &'a TestDevice,
        operands: AttentionBackwardOperands<'a, TestBuffer<f32>, f32>,
    ) -> hephaestus_core::Result<Self::PreparedBackward<'a>> {
        plan_attention_backward(&operands, false)?;
        let selected = usize::from(operands.gradients.query.is_some())
            | (usize::from(operands.gradients.key.is_some()) << 1)
            | (usize::from(operands.gradients.value.is_some()) << 2);
        BACKWARD_GRADIENTS.store(selected, Ordering::SeqCst);
        Ok(TestPreparedBackward { operands })
    }

    fn dispatch_attention_backward(
        &self,
        _device: &TestDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> hephaestus_core::Result<()> {
        let operands = &prepared.operands;
        let grad_output_values = operands
            .grad_output
            .buffer
            .values
            .lock()
            .expect("test attention output gradient lock");
        let query_values = operands
            .query
            .buffer
            .values
            .lock()
            .expect("test query buffer lock");
        let key_values = operands
            .key
            .buffer
            .values
            .lock()
            .expect("test key buffer lock");
        let value_values = operands
            .value
            .buffer
            .values
            .lock()
            .expect("test value buffer lock");
        let weight_values = operands
            .weights
            .buffer
            .values
            .lock()
            .expect("test attention weights lock");
        let grad_output = ArrayView::try_new(*operands.grad_output.layout, &grad_output_values)
            .map_err(attention_error)?;
        let query =
            ArrayView::try_new(*operands.query.layout, &query_values).map_err(attention_error)?;
        let key = ArrayView::try_new(*operands.key.layout, &key_values).map_err(attention_error)?;
        let value =
            ArrayView::try_new(*operands.value.layout, &value_values).map_err(attention_error)?;
        let weights = ArrayView::try_new(*operands.weights.layout, &weight_values)
            .map_err(attention_error)?;
        let mut query_gradient_values = operands.gradients.query.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test query gradient lock")
        });
        let mut key_gradient_values = operands.gradients.key.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test key gradient lock")
        });
        let mut value_gradient_values = operands.gradients.value.map(|gradient| {
            gradient
                .buffer
                .values
                .lock()
                .expect("test value gradient lock")
        });
        let mut query_gradient = match (query_gradient_values.as_mut(), operands.gradients.query) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "query gradient descriptor and storage disagree".into(),
                });
            }
        };
        let mut key_gradient = match (key_gradient_values.as_mut(), operands.gradients.key) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "key gradient descriptor and storage disagree".into(),
                });
            }
        };
        let mut value_gradient = match (value_gradient_values.as_mut(), operands.gradients.value) {
            (Some(values), Some(target)) => Some(
                ArrayViewMut::try_new(*target.layout, values.as_mut_slice())
                    .map_err(attention_error)?,
            ),
            (None, None) => None,
            _ => {
                return Err(HephaestusError::InvalidConfiguration {
                    message: "value gradient descriptor and storage disagree".into(),
                });
            }
        };
        scaled_dot_product_attention_backward_accumulate(
            &grad_output,
            &query,
            &key,
            &value,
            &weights,
            operands.scale,
            AttentionGradients::new(
                query_gradient.take(),
                key_gradient.take(),
                value_gradient.take(),
            ),
        )
        .map_err(attention_error)
    }
}

pub(crate) fn attention_error(error: impl std::fmt::Display) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: error.to_string(),
    }
}

// SAFETY: `TestAttentionOps` writes both forward output buffers on success and
// preserves initialized gradient values during its additive backward path.
unsafe impl AttentionProvider<f32> for TestProvider {
    type Operations = TestAttentionOps;
}
