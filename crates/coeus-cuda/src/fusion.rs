//! Coeus expression and layout adaptation for provider-owned CUDA fusion.

use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_core::{BackendError, Layout, Scalar};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use hephaestus_core::{
    CudaC, DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction,
    FusedReductionOps,
};
use hephaestus_cuda::CudaFusionOps;
use leto::LayoutDyn;

use crate::backend::{CudaBackend, CudaScalar};
use crate::storage::CudaStorage;
use crate::CudaBackendError;

/// Adapt Coeus's expression-local input names to the provider vocabulary.
struct ExpressionAdapter<'expression, T, E> {
    expression: &'expression E,
    _scalar: PhantomData<T>,
}

impl<T, E> FusedExpression<CudaC> for ExpressionAdapter<'_, T, E>
where
    T: CudaScalar,
    E: ExprNode<T, CudaBackend>,
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
) -> Result<LayoutDyn, CudaBackendError> {
    if layout.shape().len() != layout.strides().len() {
        return Err(CudaBackendError::validation(BackendError::Storage {
            operation,
            reason: format!(
                "layout shape rank {} differs from stride rank {}",
                layout.shape().len(),
                layout.strides().len()
            ),
        }));
    }
    let strides = layout
        .strides()
        .iter()
        .enumerate()
        .map(|(axis, &stride)| {
            isize::try_from(stride).map_err(|_| {
                CudaBackendError::validation(BackendError::Storage {
                    operation,
                    reason: format!("layout stride at axis {axis} exceeds isize range: {stride}"),
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayoutDyn::new(
        layout.shape().to_vec().into_boxed_slice(),
        strides.into_boxed_slice(),
        layout.offset(),
    )
    .map_err(|error| {
        CudaBackendError::validation(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })
}

fn input_views<'inputs, T: Scalar>(
    inputs: &'inputs [&'inputs Tensor<T, CudaBackend>],
    layouts: &'inputs [LayoutDyn],
) -> Vec<DynamicStridedView<'inputs, hephaestus_cuda::CudaBuffer<T>>> {
    inputs
        .iter()
        .zip(layouts)
        .map(|(tensor, layout)| DynamicStridedView::new(tensor.storage().buffer.as_ref(), layout))
        .collect()
}

/// Dispatch a fused elementwise expression through Hephaestus CUDA.
pub(crate) fn dispatch_fused<T, E>(
    expression: &E,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar,
    E: ExprNode<T, CudaBackend>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(CudaBackendError::validation(BackendError::Storage {
            operation: "fused elementwise",
            reason: "expression contains no tensor inputs".to_owned(),
        }));
    }
    let input_layouts = inputs
        .iter()
        .map(|tensor| provider_layout(tensor.layout(), "fused elementwise"))
        .collect::<Result<Vec<_>, _>>()?;
    let output_layout = provider_layout(output_layout, "fused elementwise")?;
    let input_views = input_views(&inputs, &input_layouts);
    let output_view = DynamicStridedView::new(output.buffer.as_ref(), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _scalar: PhantomData,
    };
    let device = crate::backend::try_get_cuda_device()
        .map_err(|source| CudaBackendError::dispatch("fused elementwise", source))?;
    CudaFusionOps
        .fused_elementwise_into(device, &adapter, &input_views, output_view)
        .map_err(|source| CudaBackendError::dispatch("fused elementwise", source))
}

/// Dispatch a fused expression reduction through Hephaestus CUDA.
pub(crate) fn dispatch_fused_reduce<T, E>(
    expression: &E,
    reduction: coeus_ops::ReductionOp,
    axis: usize,
    output: &mut CudaStorage<T>,
    output_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar,
    E: ExprNode<T, CudaBackend>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(CudaBackendError::validation(BackendError::Storage {
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
    let device = crate::backend::try_get_cuda_device()
        .map_err(|source| CudaBackendError::dispatch("fused reduction", source))?;
    CudaFusionOps
        .fused_reduce_into(device, &adapter, &input_views, reduction, axis, output_view)
        .map_err(|source| CudaBackendError::dispatch("fused reduction", source))
}
