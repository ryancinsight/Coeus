use super::{gradients, layouts};
use crate::{attention::provider::AttentionBackend, HephaestusProvider};
use coeus_core::{Float, Layout, Scalar, StorageMut};
use hephaestus_core::{
    plan_attention_backward, AttentionBackwardOperands, AttentionOps, AttentionScalar, StridedView,
};

const OPERATION: &str = "attention backward";

pub(in crate::attention) struct Backward<'a, B, T>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    pub grad_output: &'a B::DeviceBuffer<T>,
    pub grad_output_layout: &'a Layout,
    pub query: &'a B::DeviceBuffer<T>,
    pub query_layout: &'a Layout,
    pub key: &'a B::DeviceBuffer<T>,
    pub key_layout: &'a Layout,
    pub value: &'a B::DeviceBuffer<T>,
    pub value_layout: &'a Layout,
    pub weights: &'a B::DeviceBuffer<T>,
    pub weights_layout: &'a Layout,
    pub scale: T,
    pub grad_query: Option<(&'a mut B::DeviceBuffer<T>, &'a Layout)>,
    pub grad_key: Option<(&'a mut B::DeviceBuffer<T>, &'a Layout)>,
    pub grad_value: Option<(&'a mut B::DeviceBuffer<T>, &'a Layout)>,
}

pub(in crate::attention) fn execute<B, T>(mut request: Backward<'_, B, T>) -> Result<(), B::Error>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    let grad_output_layout = layouts::tensor(OPERATION, request.grad_output_layout)?;
    let query_layout = layouts::tensor(OPERATION, request.query_layout)?;
    let key_layout = layouts::tensor(OPERATION, request.key_layout)?;
    let value_layout = layouts::tensor(OPERATION, request.value_layout)?;
    let weights_layout = layouts::tensor(OPERATION, request.weights_layout)?;
    let grad_query_layout = request
        .grad_query
        .as_ref()
        .map(|(_, layout)| layouts::tensor(OPERATION, layout))
        .transpose()?;
    let grad_key_layout = request
        .grad_key
        .as_ref()
        .map(|(_, layout)| layouts::tensor(OPERATION, layout))
        .transpose()?;
    let grad_value_layout = request
        .grad_value
        .as_ref()
        .map(|(_, layout)| layouts::tensor(OPERATION, layout))
        .transpose()?;
    {
        let operands = AttentionBackwardOperands {
            grad_output: StridedView::new(
                B::attention_buffer(request.grad_output),
                &grad_output_layout,
            ),
            query: StridedView::new(B::attention_buffer(request.query), &query_layout),
            key: StridedView::new(B::attention_buffer(request.key), &key_layout),
            value: StridedView::new(B::attention_buffer(request.value), &value_layout),
            weights: StridedView::new(B::attention_buffer(request.weights), &weights_layout),
            scale: request.scale,
            gradients: gradients::bind(
                request
                    .grad_query
                    .as_ref()
                    .zip(grad_query_layout.as_ref())
                    .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
                request
                    .grad_key
                    .as_ref()
                    .zip(grad_key_layout.as_ref())
                    .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
                request
                    .grad_value
                    .as_ref()
                    .zip(grad_value_layout.as_ref())
                    .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
            ),
        };
        // Shared storage is detached below; preflight validates descriptors before
        // any destination changes. Dispatch then checks the detached buffer aliases.
        plan_attention_backward(&operands, false)
            .map_err(|source| B::attention_dispatch_error(OPERATION, source))?;
    }
    if let Some((buffer, _)) = request.grad_query.as_mut() {
        buffer.make_unique();
    }
    if let Some((buffer, _)) = request.grad_key.as_mut() {
        buffer.make_unique();
    }
    if let Some((buffer, _)) = request.grad_value.as_mut() {
        buffer.make_unique();
    }
    let operations = <<B as AttentionBackend<T>>::Provider as super::super::provider::AttentionProvider<T>>::Operations::default();
    operations
        .attention_backward_accumulate(
            <B::Provider as HephaestusProvider>::device(),
            AttentionBackwardOperands {
                grad_output: StridedView::new(
                    B::attention_buffer(request.grad_output),
                    &grad_output_layout,
                ),
                query: StridedView::new(B::attention_buffer(request.query), &query_layout),
                key: StridedView::new(B::attention_buffer(request.key), &key_layout),
                value: StridedView::new(B::attention_buffer(request.value), &value_layout),
                weights: StridedView::new(B::attention_buffer(request.weights), &weights_layout),
                scale: request.scale,
                gradients: gradients::bind(
                    request
                        .grad_query
                        .as_ref()
                        .zip(grad_query_layout.as_ref())
                        .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
                    request
                        .grad_key
                        .as_ref()
                        .zip(grad_key_layout.as_ref())
                        .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
                    request
                        .grad_value
                        .as_ref()
                        .zip(grad_value_layout.as_ref())
                        .map(|((buffer, _), layout)| (B::attention_buffer(buffer), layout)),
                ),
            },
        )
        .map_err(|source| B::attention_dispatch_error(OPERATION, source))
}
