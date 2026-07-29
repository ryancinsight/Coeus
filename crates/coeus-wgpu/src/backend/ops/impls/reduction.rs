use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::{BackendError, Layout};
use hephaestus_core::{CombineExpr, IdentityToken, OpIdentity};
use hephaestus_wgpu::{
    CumProdOp, CumSumOp, MaxOp, MinOp, ProdOp, ScanDirection, StridedOperand, SumOp, Wgsl,
};
use leto::Layout as LetoLayout;

fn provider_layout(layout: &Layout) -> Option<LetoLayout<2>> {
    match (layout.shape(), layout.strides()) {
        ([length], [stride]) => Some(LetoLayout::new(
            [1, *length],
            [0, isize::try_from(*stride).ok()?],
            layout.offset(),
        )),
        ([rows, columns], [row_stride, column_stride]) => Some(LetoLayout::new(
            [*rows, *columns],
            [
                isize::try_from(*row_stride).ok()?,
                isize::try_from(*column_stride).ok()?,
            ],
            layout.offset(),
        )),
        _ => None,
    }
}

fn provider_axis(
    operation: &'static str,
    layout: &Layout,
    axis: usize,
) -> Result<usize, crate::backend::WgpuBackendError> {
    if axis >= layout.ndim() {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::AxisOutOfRange {
                operation,
                axis,
                rank: layout.ndim(),
            },
        ));
    }
    Ok(if layout.ndim() == 1 { 1 } else { axis })
}

fn dispatch_scan<Op, T>(
    operation: &'static str,
    direction: ScanDirection,
    a: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError>
where
    Op: CombineExpr<Wgsl>,
    T: WgpuScalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<Wgsl>
        + OpIdentity<Op>
        + IdentityToken<Op, Wgsl>,
{
    if a_layout.shape() != c_layout.shape() {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::ShapeMismatch {
                operation,
                lhs: a_layout.shape().to_vec(),
                rhs: c_layout.shape().to_vec(),
            },
        ));
    }
    let Some(input_layout) = provider_layout(a_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation,
                rank: a_layout.ndim(),
                max_rank: 2,
            },
        ));
    };
    let Some(output_layout) = provider_layout(c_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation,
                rank: c_layout.ndim(),
                max_rank: 2,
            },
        ));
    };
    let input = StridedOperand {
        buffer: a.buffer.as_ref(),
        layout: &input_layout,
    };
    let output = StridedOperand {
        buffer: c.buffer.as_ref(),
        layout: &output_layout,
    };
    let device = &crate::backend::get_wgpu_context().hephaestus_device;
    let provider_axis = provider_axis(operation, a_layout, axis)?;
    let result = hephaestus_wgpu::scan_axis_into::<Op, T>(
        device,
        input,
        provider_axis,
        direction,
        output,
        hephaestus_core::BlockWidth::DEFAULT,
    );
    result.map_err(|source| crate::backend::WgpuBackendError::dispatch(operation, source))
}

fn dispatch_reduction<T>(
    op: coeus_ops::ReductionOp,
    a: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError>
where
    T: WgpuScalar
        + hephaestus_wgpu::DialectScalar<Wgsl>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, Wgsl>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Wgsl>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, Wgsl>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>,
{
    let Some(input_layout) = provider_layout(a_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation: "reduction",
                rank: a_layout.ndim(),
                max_rank: 2,
            },
        ));
    };
    let Some(output_layout) = provider_layout(c_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation: "reduction",
                rank: c_layout.ndim(),
                max_rank: 2,
            },
        ));
    };
    let input = StridedOperand {
        buffer: a.buffer.as_ref(),
        layout: &input_layout,
    };
    let output = StridedOperand {
        buffer: c.buffer.as_ref(),
        layout: &output_layout,
    };
    let provider_axis = provider_axis("reduction", a_layout, axis)?;
    let device = &crate::backend::get_wgpu_context().hephaestus_device;
    let width = hephaestus_core::BlockWidth::DEFAULT;
    let result = match op {
        coeus_ops::ReductionOp::Sum => {
            hephaestus_wgpu::sum_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Mean => {
            hephaestus_wgpu::mean_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Prod => {
            hephaestus_wgpu::prod_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Min => {
            hephaestus_wgpu::min_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Max => {
            hephaestus_wgpu::max_axis_into(device, input, provider_axis, output, width)
        }
    };
    result.map_err(|source| crate::backend::WgpuBackendError::dispatch("reduction", source))
}

impl<
    T: WgpuScalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<Wgsl>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, Wgsl>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, Wgsl>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, Wgsl>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, Wgsl>
        + OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, Wgsl>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, Wgsl>,
> coeus_ops::ReductionOps<T> for WgpuBackend
{
    #[inline]
    fn reduce(
        &self,
        op: coeus_ops::ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch_reduction(op, a, a_layout, axis, c, c_layout)
    }

    #[inline]
    fn cumsum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        dispatch_scan::<CumSumOp, T>(
            "cumsum",
            ScanDirection::Forward,
            a,
            a_layout,
            axis,
            c,
            c_layout,
        )
    }

    #[inline]
    fn suffix_sum(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        dispatch_scan::<CumSumOp, T>(
            "suffix_sum",
            ScanDirection::Reverse,
            a,
            a_layout,
            axis,
            c,
            c_layout,
        )
    }

    #[inline]
    fn cumprod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        dispatch_scan::<CumProdOp, T>(
            "cumprod",
            ScanDirection::Forward,
            a,
            a_layout,
            axis,
            c,
            c_layout,
        )
    }

    #[inline]
    fn suffix_prod(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        T: leto_ops::Scalar,
    {
        dispatch_scan::<CumProdOp, T>(
            "suffix_prod",
            ScanDirection::Reverse,
            a,
            a_layout,
            axis,
            c,
            c_layout,
        )
    }
}
