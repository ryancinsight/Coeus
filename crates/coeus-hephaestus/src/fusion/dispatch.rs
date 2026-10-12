//! Coeus expression and layout adaptation for provider-owned fusion.
//!
//! One generic dispatch serves every [`FusionProvider`](super::FusionProvider):
//! the expression adapter, layout validation, and empty-axis semantics are
//! provider-independent; only the operation bundle and kernel dialect vary.

use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;

use coeus_core::{BackendError, ComputeBackend, Layout, Scalar, StorageMut};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use hephaestus_core::{
    ComputeDevice, DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction,
    FusedReductionOps,
};
use leto::LayoutDyn;

use super::FusionProvider;
use crate::{HephaestusBackend, HephaestusBackendError, HephaestusProvider, HephaestusStorage};

type Buffer<P, T> = <<P as HephaestusProvider>::Device as ComputeDevice>::Buffer<T>;

/// Adapt Coeus's expression-local input names to the provider vocabulary.
struct ExpressionAdapter<'expression, P, T, E> {
    expression: &'expression E,
    _provider: PhantomData<(P, T)>,
}

impl<P, T, E> FusedExpression<P::Dialect> for ExpressionAdapter<'_, P, T, E>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
    E: ExprNode<T, HephaestusBackend<P>>,
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
) -> Result<LayoutDyn, HephaestusBackendError> {
    if layout.shape().len() != layout.strides().len() {
        return Err(HephaestusBackendError::from(
            BackendError::LayoutRankMismatch {
                operation,
                lhs: layout.shape().len(),
                rhs: layout.strides().len(),
            },
        ));
    }
    let strides = layout
        .strides()
        .iter()
        .enumerate()
        .map(|(axis, &stride)| {
            isize::try_from(stride).map_err(|_| {
                HephaestusBackendError::from(BackendError::Storage {
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
        HephaestusBackendError::from(BackendError::Storage {
            operation,
            reason: error.to_string(),
        })
    })
}

fn input_views<'inputs, P, T>(
    inputs: &'inputs [&'inputs Tensor<T, HephaestusBackend<P>>],
    layouts: &'inputs [LayoutDyn],
) -> Vec<DynamicStridedView<'inputs, Buffer<P, T>>>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
{
    inputs
        .iter()
        .zip(layouts)
        .map(|(tensor, layout)| DynamicStridedView::new(tensor.storage().buffer(), layout))
        .collect()
}

fn dispatch_fused<P, T, E>(
    expression: &E,
    output: &mut HephaestusStorage<P, T>,
    output_layout: &Layout,
) -> Result<(), HephaestusBackendError>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
    E: ExprNode<T, HephaestusBackend<P>>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(HephaestusBackendError::from(BackendError::Storage {
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
    output.make_unique();
    let output_view = DynamicStridedView::new(output.buffer(), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _provider: PhantomData,
    };
    let device = P::try_device()
        .map_err(|source| HephaestusBackendError::device("fused elementwise", source))?;
    P::Operations::default()
        .fused_elementwise_into(device, &adapter, &input_views, output_view)
        .map_err(|source| HephaestusBackendError::device("fused elementwise", source))
}

fn dispatch_fused_reduce<P, T, E>(
    expression: &E,
    reduction: coeus_ops::ReductionOp,
    axis: usize,
    output: &mut HephaestusStorage<P, T>,
    output_layout: &Layout,
) -> Result<(), HephaestusBackendError>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
    E: ExprNode<T, HephaestusBackend<P>>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(HephaestusBackendError::from(BackendError::Storage {
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
    output.make_unique();
    let output_view = DynamicStridedView::new(output.buffer(), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _provider: PhantomData,
    };
    let reduction = match reduction {
        coeus_ops::ReductionOp::Sum => FusedReduction::Sum,
        coeus_ops::ReductionOp::Prod => FusedReduction::Product,
        coeus_ops::ReductionOp::Mean => FusedReduction::Mean,
        coeus_ops::ReductionOp::Max => FusedReduction::Maximum,
        coeus_ops::ReductionOp::Min => FusedReduction::Minimum,
    };
    let device = P::try_device()
        .map_err(|source| HephaestusBackendError::device("fused reduction", source))?;
    P::Operations::default()
        .fused_reduce_into(device, &adapter, &input_views, reduction, axis, output_view)
        .map_err(|source| HephaestusBackendError::device("fused reduction", source))
}

fn checked_numel(
    operation: &'static str,
    shape: &[usize],
) -> Result<usize, HephaestusBackendError> {
    shape.iter().try_fold(1usize, |numel, &dimension| {
        numel
            .checked_mul(dimension)
            .ok_or(HephaestusBackendError::from(BackendError::Overflow {
                operation,
                reason: "output element-count arithmetic overflow",
            }))
    })
}

/// Evaluate a fused element-wise expression on the provider device.
///
/// The provider must be live, the expression layout valid, and the generated
/// kernel launch successful. Provider failure is returned to the caller; this
/// API never changes execution backends after a dispatch has been selected.
///
/// # Errors
///
/// Returns [`HephaestusBackendError`] when the expression has no tensor input,
/// or the provider, generated kernel, or launch ABI rejects the operation.
pub fn evaluate_fused<P, T, E>(
    expr: &E,
) -> Result<Tensor<T, HephaestusBackend<P>>, HephaestusBackendError>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
    E: ExprNode<T, HephaestusBackend<P>>,
{
    const OPERATION: &str = "fused elementwise";

    let out_shape = expr.shape()?.ok_or_else(|| {
        HephaestusBackendError::from(BackendError::Storage {
            operation: OPERATION,
            reason: "expression has no tensor input from which to derive its shape".to_string(),
        })
    })?;
    let out_layout = Layout::new(out_shape);
    let mut out_storage = HephaestusStorage::<P, _>::new(out_layout.numel());

    dispatch_fused(expr, &mut out_storage, &out_layout)?;

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}

/// Evaluate a fused reduction along an axis on the provider device.
///
/// An empty reduction axis fills the identity for sum (zero) and product
/// (one); empty mean, maximum, and minimum reductions are undefined and
/// rejected.
///
/// # Errors
///
/// Returns [`HephaestusBackendError`] when the expression has no tensor input,
/// the axis is invalid, an empty axis is used with mean, maximum, or minimum,
/// or the provider, generated kernel, or launch ABI rejects the operation.
pub fn evaluate_fused_reduce<P, T, E>(
    expr: &E,
    op: coeus_ops::ReductionOp,
    axis: usize,
) -> Result<Tensor<T, HephaestusBackend<P>>, HephaestusBackendError>
where
    P: FusionProvider<T>,
    T: Scalar + leto_ops::Scalar,
    E: ExprNode<T, HephaestusBackend<P>>,
{
    const OPERATION: &str = "fused reduction";

    let mut inputs = Vec::new();
    expr.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(HephaestusBackendError::from(BackendError::Storage {
            operation: OPERATION,
            reason: "expression contains no tensor inputs".to_string(),
        }));
    }
    let expr_shape = expr.shape()?.ok_or_else(|| {
        HephaestusBackendError::from(BackendError::Storage {
            operation: OPERATION,
            reason: "expression has no tensor input from which to derive its shape".to_string(),
        })
    })?;

    let out_rank = expr_shape.len();
    let axis_len = *expr_shape.get(axis).ok_or({
        HephaestusBackendError::from(BackendError::AxisOutOfRange {
            operation: OPERATION,
            axis,
            rank: out_rank,
        })
    })?;
    let mut out_shape = expr_shape.clone();
    let output_axis = out_shape.get_mut(axis).ok_or({
        HephaestusBackendError::from(BackendError::AxisOutOfRange {
            operation: OPERATION,
            axis,
            rank: out_rank,
        })
    })?;
    *output_axis = 1;
    let out_layout = Layout::new(out_shape.clone());
    let out_numel = checked_numel(OPERATION, out_layout.shape())?;
    let mut out_storage = HephaestusStorage::<P, _>::new(out_numel);

    if axis_len == 0 {
        let identity = match op {
            coeus_ops::ReductionOp::Sum => T::ZERO,
            coeus_ops::ReductionOp::Prod => T::ONE,
            coeus_ops::ReductionOp::Mean
            | coeus_ops::ReductionOp::Max
            | coeus_ops::ReductionOp::Min => {
                return Err(BackendError::EmptyReduction {
                    operation: OPERATION,
                    reduction: op,
                }
                .into());
            }
        };
        HephaestusBackend::<P>::new().fill(&mut out_storage, identity);
        return Ok(Tensor::from_raw_parts(out_storage, out_layout));
    }

    dispatch_fused_reduce(expr, op, axis, &mut out_storage, &out_layout)?;

    Ok(Tensor::from_raw_parts(out_storage, out_layout))
}
