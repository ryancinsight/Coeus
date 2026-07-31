use super::{gradients, layouts};
use crate::{attention::provider::AttentionBackend, HephaestusProvider};
use coeus_core::{Float, Layout, Scalar};
use hephaestus_core::{AttentionBackwardOperands, AttentionOps, AttentionScalar, StridedView};
use leto::Layout as LetoLayout;

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

type ProviderBuffer<B, T> = <<<B as AttentionBackend<T>>::Provider as HephaestusProvider>::Device as hephaestus_core::ComputeDevice>::Buffer<T>;
type ProjectedGradient<'a, B, T> = Option<(&'a ProviderBuffer<B, T>, LetoLayout<3>)>;

fn project_gradient<'a, B, T>(
    destination: Option<(&'a mut B::DeviceBuffer<T>, &'a Layout)>,
) -> Result<ProjectedGradient<'a, B, T>, B::Error>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    destination
        .map(|(buffer, layout)| {
            Ok((
                B::attention_buffer(&*buffer),
                layouts::tensor(OPERATION, layout)?,
            ))
        })
        .transpose()
}

pub(in crate::attention) fn execute<B, T>(request: Backward<'_, B, T>) -> Result<(), B::Error>
where
    B: AttentionBackend<T>,
    T: Scalar + Float + AttentionScalar,
{
    let grad_output_layout = layouts::tensor(OPERATION, request.grad_output_layout)?;
    let query_layout = layouts::tensor(OPERATION, request.query_layout)?;
    let key_layout = layouts::tensor(OPERATION, request.key_layout)?;
    let value_layout = layouts::tensor(OPERATION, request.value_layout)?;
    let weights_layout = layouts::tensor(OPERATION, request.weights_layout)?;
    let grad_query = project_gradient::<B, T>(request.grad_query)?;
    let grad_key = project_gradient::<B, T>(request.grad_key)?;
    let grad_value = project_gradient::<B, T>(request.grad_value)?;
    let gradients = gradients::bind(
        grad_query
            .as_ref()
            .map(|(buffer, layout)| (*buffer, layout)),
        grad_key.as_ref().map(|(buffer, layout)| (*buffer, layout)),
        grad_value
            .as_ref()
            .map(|(buffer, layout)| (*buffer, layout)),
    );
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
                gradients,
            },
        )
        .map_err(|source| B::attention_dispatch_error(OPERATION, source))
}
