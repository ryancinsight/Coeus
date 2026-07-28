use crate::CudaBackendError;
use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::{BackendError, Layout};
use hephaestus_cuda::StridedOperand;
use hephaestus_cuda::{
    CombineExpr, CumProdOp, CumSumOp, IdentityToken, MaxOp, MinOp, OpIdentity, ProdOp,
    ScanDirection, SumOp,
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
) -> Result<usize, CudaBackendError> {
    if axis >= layout.ndim() {
        return Err(CudaBackendError::validation(BackendError::AxisOutOfRange {
            operation,
            axis,
            rank: layout.ndim(),
        }));
    }
    Ok(if layout.ndim() == 1 { 1 } else { axis })
}

fn dispatch_scan<Op, T>(
    operation: &'static str,
    direction: ScanDirection,
    a: &crate::backend::CudaStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::CudaStorage<T>,
    c_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    Op: CombineExpr<hephaestus_cuda::CudaC>,
    T: CudaScalar
        + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>
        + OpIdentity<Op>
        + IdentityToken<Op, hephaestus_cuda::CudaC>,
{
    if a_layout.shape() != c_layout.shape() {
        return Err(CudaBackendError::InvalidLayout {
            operation,
            reason: "input and output shapes must match",
        });
    }
    let Some(input_layout) = provider_layout(a_layout) else {
        return Err(CudaBackendError::UnsupportedRank {
            operation,
            rank: a_layout.ndim(),
            max_rank: 2,
        });
    };
    let Some(output_layout) = provider_layout(c_layout) else {
        return Err(CudaBackendError::UnsupportedRank {
            operation,
            rank: c_layout.ndim(),
            max_rank: 2,
        });
    };
    let input = StridedOperand {
        buffer: a.buffer.as_ref(),
        layout: &input_layout,
    };
    let output = StridedOperand {
        buffer: c.buffer.as_ref(),
        layout: &output_layout,
    };
    let provider_axis = provider_axis(operation, a_layout, axis)?;
    let device = crate::backend::get_cuda_device();
    hephaestus_cuda::scan_axis_into::<Op, T>(
        device,
        input,
        provider_axis,
        direction,
        output,
        hephaestus_cuda::BlockWidth::DEFAULT,
    )
    .map_err(|source| CudaBackendError::dispatch(operation, source))
}

fn dispatch_reduction<T>(
    op: coeus_ops::ReductionOp,
    a: &crate::backend::CudaStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::CudaStorage<T>,
    c_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar
        + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, hephaestus_cuda::CudaC>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, hephaestus_cuda::CudaC>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, hephaestus_cuda::CudaC>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, hephaestus_cuda::CudaC>,
{
    let Some(input_layout) = provider_layout(a_layout) else {
        return Err(CudaBackendError::UnsupportedRank {
            operation: "reduction",
            rank: a_layout.ndim(),
            max_rank: 2,
        });
    };
    let Some(output_layout) = provider_layout(c_layout) else {
        return Err(CudaBackendError::UnsupportedRank {
            operation: "reduction",
            rank: c_layout.ndim(),
            max_rank: 2,
        });
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
    let device = crate::backend::get_cuda_device();
    let width = hephaestus_cuda::BlockWidth::DEFAULT;
    let result = match op {
        coeus_ops::ReductionOp::Sum => {
            hephaestus_cuda::sum_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Mean => {
            hephaestus_cuda::mean_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Prod => {
            hephaestus_cuda::prod_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Min => {
            hephaestus_cuda::min_axis_into(device, input, provider_axis, output, width)
        }
        coeus_ops::ReductionOp::Max => {
            hephaestus_cuda::max_axis_into(device, input, provider_axis, output, width)
        }
    };
    result.map_err(|source| CudaBackendError::dispatch("reduction", source))
}

impl<
    T: CudaScalar
        + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, hephaestus_cuda::CudaC>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, hephaestus_cuda::CudaC>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, hephaestus_cuda::CudaC>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, hephaestus_cuda::CudaC>
        + OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, hephaestus_cuda::CudaC>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, hephaestus_cuda::CudaC>,
> coeus_ops::ReductionOps<T> for CudaBackend
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
