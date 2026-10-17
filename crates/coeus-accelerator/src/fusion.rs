//! Vendor-neutral fused expression dispatch (ADR 0082).
//!
//! One generic implementation serves every GPU vendor: the expression
//! adapter, layout adaptation, input projection, and the
//! acquire-validate-dispatch sequence are identical across backends and
//! differ only in the [`FusionBackend`] bundle each vendor provides.
//! Monomorphization instantiates one copy per vendor backend; no vtable,
//! no macro-generated duplicate, no CPU fallback.

use crate::seam::HephaestusSeamBackend;
use coeus_core::{BackendError, Layout, Scalar};
use coeus_ops::fuse::ExprNode;
use coeus_tensor::Tensor;
use hephaestus_core::{
    DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction, FusedReductionOps,
    KernelDialect,
};
use leto::LayoutDyn;
use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Vendor fusion bundle: the device seam plus the vendor's fusion operation
/// set and typed validation mapping.
///
/// # Safety
///
/// Implementations uphold the [`HephaestusSeamBackend`] contract; the shared
/// dispatch additionally requires that a successful `fused_*_into` call
/// initializes every logical output element.
pub unsafe trait FusionBackend<T>: HephaestusSeamBackend
where
    T: Scalar,
{
    /// Shader dialect authored by the vendor fusion operations.
    type Dialect: KernelDialect;
    /// Monomorphized Hephaestus fusion operations for the vendor device.
    type FusionOps: FusedElementwiseOps<Self::Device, T, Dialect = Self::Dialect>
        + FusedReductionOps<Self::Device, T, Dialect = Self::Dialect>
        + Default;

    /// Map a shared validation failure into the vendor error type without
    /// stringifying the typed failure.
    fn validation_error(error: BackendError) -> Self::Error;
}

/// Zero-sized adapter bridging [`ExprNode`] trees to [`FusedExpression`].
struct ExpressionAdapter<'expression, T: Scalar, B: FusionBackend<T>, E: ExprNode<T, B>> {
    expression: &'expression E,
    _scalar: PhantomData<T>,
    _backend: PhantomData<B>,
}

impl<T, B, E> FusedExpression<B::Dialect> for ExpressionAdapter<'_, T, B, E>
where
    T: Scalar,
    B: FusionBackend<T>,
    E: ExprNode<T, B>,
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

/// Project a [`Layout`] into a dynamic descriptor, reporting rank and stride
/// failures with the precise typed variants on every vendor.
fn provider_layout(layout: &Layout, operation: &'static str) -> Result<LayoutDyn, BackendError> {
    if layout.shape().len() != layout.strides().len() {
        return Err(BackendError::LayoutRankMismatch {
            operation,
            lhs: layout.shape().len(),
            rhs: layout.strides().len(),
        });
    }
    let mut strides = Vec::with_capacity(layout.strides().len());
    for &stride in layout.strides() {
        strides.push(isize::try_from(stride).map_err(|_| BackendError::Overflow {
            operation,
            reason: "layout stride exceeds isize range",
        })?);
    }
    LayoutDyn::new(
        layout.shape().to_vec().into_boxed_slice(),
        strides.into_boxed_slice(),
        layout.offset(),
    )
    .map_err(|error| BackendError::Storage {
        operation,
        reason: error.to_string(),
    })
}

/// Project input tensors into dynamic strided views over vendor buffers.
fn input_views<'inputs, T, B>(
    inputs: &'inputs [&'inputs Tensor<T, B>],
    layouts: &'inputs [LayoutDyn],
) -> Vec<DynamicStridedView<'inputs, <B::Device as hephaestus_core::ComputeDevice>::Buffer<T>>>
where
    T: Scalar,
    B: FusionBackend<T>,
{
    inputs
        .iter()
        .zip(layouts)
        .map(|(tensor, layout)| DynamicStridedView::new(B::buffer(tensor.storage()), layout))
        .collect()
}

/// Dispatch a fused elementwise expression through the vendor device.
///
/// # Errors
///
/// Returns the vendor-mapped validation failure or the typed device
/// acquisition/dispatch failure without a CPU fallback.
pub fn dispatch_fused<T, B, E>(
    expression: &E,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar,
    B: FusionBackend<T>,
    E: ExprNode<T, B>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(B::validation_error(BackendError::Storage {
            operation: "fused elementwise",
            reason: "expression contains no tensor inputs".to_owned(),
        }));
    }
    let input_layouts = inputs
        .iter()
        .map(|tensor| {
            provider_layout(tensor.layout(), "fused elementwise").map_err(B::validation_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_layout =
        provider_layout(output_layout, "fused elementwise").map_err(B::validation_error)?;
    let input_views = input_views(&inputs, &input_layouts);
    coeus_core::StorageMut::make_unique(output)?;
    let output_view = DynamicStridedView::new(B::buffer(output), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _scalar: PhantomData,
        _backend: PhantomData,
    };
    let device =
        B::try_device().map_err(|source| B::dispatch_error("fused elementwise", source))?;
    B::FusionOps::default()
        .fused_elementwise_into(device, &adapter, &input_views, output_view)
        .map_err(|source| B::dispatch_error("fused elementwise", source))
}

/// Dispatch a fused expression reduction through the vendor device.
///
/// # Errors
///
/// Returns the vendor-mapped validation failure or the typed device
/// acquisition/dispatch failure without a CPU fallback.
pub fn dispatch_fused_reduce<T, B, E>(
    expression: &E,
    reduction: FusedReduction,
    axis: usize,
    output: &mut B::DeviceBuffer<T>,
    output_layout: &Layout,
) -> Result<(), B::Error>
where
    T: Scalar,
    B: FusionBackend<T>,
    E: ExprNode<T, B>,
{
    let mut inputs = Vec::new();
    expression.collect_inputs(&mut inputs);
    if inputs.is_empty() {
        return Err(B::validation_error(BackendError::Storage {
            operation: "fused reduction",
            reason: "expression contains no tensor inputs".to_owned(),
        }));
    }
    let input_layouts = inputs
        .iter()
        .map(|tensor| {
            provider_layout(tensor.layout(), "fused reduction").map_err(B::validation_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_layout =
        provider_layout(output_layout, "fused reduction").map_err(B::validation_error)?;
    let input_views = input_views(&inputs, &input_layouts);
    coeus_core::StorageMut::make_unique(output)?;
    let output_view = DynamicStridedView::new(B::buffer(output), &output_layout);
    let adapter = ExpressionAdapter {
        expression,
        _scalar: PhantomData,
        _backend: PhantomData,
    };
    let device = B::try_device().map_err(|source| B::dispatch_error("fused reduction", source))?;
    B::FusionOps::default()
        .fused_reduce_into(device, &adapter, &input_views, reduction, axis, output_view)
        .map_err(|source| B::dispatch_error("fused reduction", source))
}

#[cfg(test)]
mod tests {
    use super::provider_layout;
    use coeus_core::BackendError;

    #[test]
    fn provider_layout_reports_rank_mismatch_with_the_typed_variant() {
        let layout = coeus_core::Layout::from_shape_strides(vec![2, 3].into(), vec![1].into(), 0);
        let error = match provider_layout(&layout, "fused elementwise") {
            Ok(_) => panic!("mismatched ranks must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            BackendError::LayoutRankMismatch {
                operation: "fused elementwise",
                lhs: 2,
                rhs: 1,
            }
        );
    }

    #[test]
    fn provider_layout_reports_stride_overflow_with_the_typed_variant() {
        let layout =
            coeus_core::Layout::from_shape_strides(vec![2].into(), vec![usize::MAX].into(), 0);
        let error = match provider_layout(&layout, "fused reduction") {
            Ok(_) => panic!("overflowing strides must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            BackendError::Overflow {
                operation: "fused reduction",
                reason: "layout stride exceeds isize range",
            }
        );
    }
}
