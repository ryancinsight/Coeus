use super::{layouts, masks};
use crate::{attention::provider::AttentionBackend, HephaestusProvider};
use coeus_core::{Float, Layout, Scalar};
use hephaestus_core::{AttentionForwardOperands, AttentionOps, AttentionScalar, StridedView};

pub(in crate::attention) struct Forward<'a, B, T>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    pub query: &'a B::DeviceBuffer<T>,
    pub query_layout: &'a Layout,
    pub key: &'a B::DeviceBuffer<T>,
    pub key_layout: &'a Layout,
    pub value: &'a B::DeviceBuffer<T>,
    pub value_layout: &'a Layout,
    pub key_padding_mask: Option<&'a B::DeviceBuffer<T>>,
    pub key_padding_mask_layout: Option<&'a Layout>,
    pub is_causal: bool,
    pub scale: T,
    pub output: &'a mut B::DeviceBuffer<T>,
    pub output_layout: &'a Layout,
    pub weights: &'a mut B::DeviceBuffer<T>,
    pub weights_layout: &'a Layout,
}

pub(in crate::attention) fn execute<B, T>(request: Forward<'_, B, T>) -> Result<(), B::Error>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    const OPERATION: &str = "attention forward";
    let query_layout = layouts::tensor(OPERATION, request.query_layout)?;
    let key_layout = layouts::tensor(OPERATION, request.key_layout)?;
    let value_layout = layouts::tensor(OPERATION, request.value_layout)?;
    let output_layout = layouts::tensor(OPERATION, request.output_layout)?;
    let weights_layout = layouts::tensor(OPERATION, request.weights_layout)?;
    let mask_layout = request
        .key_padding_mask_layout
        .map(|layout| layouts::keep_mask(OPERATION, layout))
        .transpose()?;
    let mask = masks::bind(
        OPERATION,
        request.key_padding_mask.map(B::attention_buffer),
        mask_layout.as_ref(),
        query_layout.shape[0],
        request.is_causal,
    )?;
    let operations = <<B as AttentionBackend<T>>::Provider as super::super::provider::AttentionProvider<T>>::Operations::default();
    operations
        .attention_forward_into(
            <B::Provider as HephaestusProvider>::device(),
            AttentionForwardOperands {
                query: StridedView::new(B::attention_buffer(request.query), &query_layout),
                key: StridedView::new(B::attention_buffer(request.key), &key_layout),
                value: StridedView::new(B::attention_buffer(request.value), &value_layout),
                mask,
                scale: request.scale,
                output: StridedView::new(B::attention_buffer(&*request.output), &output_layout),
                weights: StridedView::new(B::attention_buffer(&*request.weights), &weights_layout),
            },
        )
        .map_err(|source| B::attention_dispatch_error(OPERATION, source))
}
