use crate::CudaBackendError;
use crate::backend::{CudaBackend, CudaScalar};
use coeus_core::Layout;
use hephaestus_cuda::StridedOperand;
use hephaestus_cuda::{IdentityToken, OpIdentity};
use leto::Layout as LetoLayout;

fn rank2_layout(layout: &Layout) -> Option<LetoLayout<2>> {
    let [rows, columns] = layout.shape() else {
        return None;
    };
    let [row_stride, column_stride] = layout.strides() else {
        return None;
    };
    Some(LetoLayout::new(
        [*rows, *columns],
        [
            isize::try_from(*row_stride).ok()?,
            isize::try_from(*column_stride).ok()?,
        ],
        layout.offset(),
    ))
}

fn dispatch_scan<T>(
    reverse: bool,
    a: &crate::backend::CudaStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::CudaStorage<T>,
    c_layout: &Layout,
) -> Result<(), CudaBackendError>
where
    T: CudaScalar
        + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>
        + OpIdentity<hephaestus_cuda::CumSumOp>
        + IdentityToken<hephaestus_cuda::CumSumOp, hephaestus_cuda::CudaC>,
{
    let operation = if reverse { "suffix_sum" } else { "cumsum" };
    if a_layout.shape() != c_layout.shape() {
        return Err(CudaBackendError::InvalidLayout {
            operation,
            reason: "input and output shapes must match",
        });
    }
    let Some(input_layout) = rank2_layout(a_layout) else {
        return Err(CudaBackendError::UnsupportedRank {
            operation,
            rank: a_layout.ndim(),
            max_rank: 2,
        });
    };
    let Some(output_layout) = rank2_layout(c_layout) else {
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
    let device = crate::backend::get_cuda_device();
    let result = if reverse {
        hephaestus_cuda::suffix_sum_into(
            device,
            input,
            axis,
            output,
            hephaestus_cuda::BlockWidth::DEFAULT,
        )
    } else {
        hephaestus_cuda::cumsum_into(
            device,
            input,
            axis,
            output,
            hephaestus_cuda::BlockWidth::DEFAULT,
        )
    };
    result.map_err(|source| CudaBackendError::dispatch(operation, source))
}

impl<
    T: CudaScalar
        + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>
        + OpIdentity<hephaestus_cuda::CumSumOp>
        + IdentityToken<hephaestus_cuda::CumSumOp, hephaestus_cuda::CudaC>,
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
        self.cuda_reduce(op, a, a_layout, axis, c, c_layout)
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
        dispatch_scan(false, a, a_layout, axis, c, c_layout)
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
        dispatch_scan(true, a, a_layout, axis, c, c_layout)
    }
}
