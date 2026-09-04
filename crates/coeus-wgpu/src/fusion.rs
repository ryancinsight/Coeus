//! Coeus expression adaptation into the provider-owned WGPU fusion seam.

use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_core::{BackendError, Layout, Scalar};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use hephaestus_core::{
    DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction, FusedReductionOps,
    Wgsl,
};
use hephaestus_wgpu::WgpuFusionOps;
use leto::LayoutDyn;

use crate::backend::{WgpuBackend, WgpuBackendError, WgpuScalar};
use crate::storage::WgpuStorage;

/// Adapt a Coeus expression's input names to the provider's canonical names.
///
/// Coeus expression nodes predate the provider seam and emit `val_N` locals.
/// The adapter translates only that consumer naming convention; Hephaestus
/// remains responsible for the WGSL wrapper, metadata, pipeline cache, and
/// command submission.
struct ExpressionAdapter<'expression, T, E> {
    expression: &'expression E,
    _scalar: PhantomData<T>,
}

impl<T, E> FusedExpression<Wgsl> for ExpressionAdapter<'_, T, E>
where
    T: WgpuScalar,
    E: ExprNode<T, WgpuBackend>,
{
    fn source(&self) -> Cow<'_, str> {
        let mut inputs = Vec::new();
        self.expression.collect_inputs(&mut inputs);
        let input_map = inputs
            .iter()
            .enumerate()
            .map(|(index, tensor)| (std::ptr::from_ref(*tensor), index))
            .collect::<HashMap<_, _>>();
        Cow::Owned(
            self.expression
                .to_shader_expr(&input_map)
                .replace("val_", "input_"),
        )
    }
}

fn provider_layout(
    layout: &Layout,
    operation: &'static str,
) -> Result<LayoutDyn, WgpuBackendError> {
    if layout.shape().len() != layout.strides().len() {
        return Err(WgpuBackendError::Layout(
            crate::backend::LayoutError::RankMismatch {
                shape_rank: layout.shape().len(),
                stride_rank: layout.strides().len(),
            },
        ));
    }
    let mut strides = Vec::with_capacity(layout.strides().len());
    for (axis, &stride) in layout.strides().iter().enumerate() {
        strides.push(isize::try_from(stride).map_err(|_| {
            WgpuBackendError::Layout(crate::backend::LayoutError::SignedStrideOutOfRange {
                axis,
                value: stride,
            })
        })?);
    }
    LayoutDyn::new(
        layout.shape().to_vec().into_boxed_slice(),
        strides.into_boxed_slice(),
        layout.offset(),
    )
    .map_err(|error| {
        WgpuBackendError::Validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })
}

fn input_views<'inputs, T: Scalar>(
    inputs: &'inputs [&'inputs Tensor<T, WgpuBackend>],
    layouts: &'inputs [LayoutDyn],
) -> Vec<DynamicStridedView<'inputs, hephaestus_wgpu::WgpuBuffer<T>>> {
    inputs
        .iter()
        .zip(layouts)
        .map(|(tensor, layout)| DynamicStridedView::new(tensor.storage().buffer.as_ref(), layout))
        .collect()
}

/// Dispatch a fused elementwise expression through Hephaestus.
pub(crate) fn dispatch_fused<T, E>(
    expression: &E,
    output: &mut WgpuStorage<T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar,
    E: ExprNode<T, WgpuBackend>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(WgpuBackendError::Validation(BackendError::Storage {
            operation: "fused expression",
            reason: "expression contains no tensor inputs".to_owned(),
        }));
    }
    let input_layouts = inputs
        .iter()
        .map(|tensor| provider_layout(tensor.layout(), "fused expression"))
        .collect::<Result<Vec<_>, _>>()?;
    let output_layout = provider_layout(output_layout, "fused expression")?;
    let input_views = input_views(&inputs, &input_layouts);
    let output_view = DynamicStridedView::new(output.buffer.as_ref(), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _scalar: PhantomData,
    };
    let device = &crate::backend::get_wgpu_context().hephaestus_device;
    WgpuFusionOps
        .fused_elementwise_into(device, &adapter, &input_views, output_view)
        .map_err(|source| WgpuBackendError::dispatch("fused elementwise", source))
}

/// Dispatch a fused expression reduction through Hephaestus.
pub(crate) fn dispatch_fused_reduce<T, E>(
    expression: &E,
    reduction: coeus_ops::ReductionOp,
    axis: usize,
    output: &mut WgpuStorage<T>,
    output_layout: &Layout,
) -> Result<(), WgpuBackendError>
where
    T: WgpuScalar,
    E: ExprNode<T, WgpuBackend>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(WgpuBackendError::Validation(BackendError::Storage {
            operation: "fused reduction",
            reason: "expression contains no tensor inputs".to_owned(),
        }));
    }
    let input_layouts = inputs
        .iter()
        .map(|tensor| provider_layout(tensor.layout(), "fused reduction"))
        .collect::<Result<Vec<_>, _>>()?;
    let output_layout = provider_layout(output_layout, "fused reduction")?;
    let input_views = input_views(&inputs, &input_layouts);
    let output_view = DynamicStridedView::new(output.buffer.as_ref(), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _scalar: PhantomData,
    };
    let reduction = match reduction {
        coeus_ops::ReductionOp::Sum => FusedReduction::Sum,
        coeus_ops::ReductionOp::Prod => FusedReduction::Product,
        coeus_ops::ReductionOp::Mean => FusedReduction::Mean,
        coeus_ops::ReductionOp::Max => FusedReduction::Maximum,
        coeus_ops::ReductionOp::Min => FusedReduction::Minimum,
    };
    let device = &crate::backend::get_wgpu_context().hephaestus_device;
    WgpuFusionOps
        .fused_reduce_into(device, &adapter, &input_views, reduction, axis, output_view)
        .map_err(|source| WgpuBackendError::dispatch("fused reduction", source))
}
