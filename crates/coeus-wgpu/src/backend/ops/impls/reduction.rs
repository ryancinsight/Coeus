use super::super::reduction;
use crate::backend::{WgpuBackend, WgpuScalar};
use coeus_core::{BackendError, Layout};
use hephaestus_core::{IdentityToken, OpIdentity};
use hephaestus_wgpu::StridedOperand;
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
    a: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    axis: usize,
    c: &mut crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<(), crate::backend::WgpuBackendError>
where
    T: WgpuScalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>
        + OpIdentity<hephaestus_wgpu::CumSumOp>
        + IdentityToken<hephaestus_wgpu::CumSumOp, hephaestus_wgpu::Wgsl>,
{
    if a_layout.shape() != c_layout.shape() {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::ShapeMismatch {
                operation: if reverse { "suffix_sum" } else { "cumsum" },
                lhs: a_layout.shape().to_vec(),
                rhs: c_layout.shape().to_vec(),
            },
        ));
    }
    let Some(input_layout) = rank2_layout(a_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation: if reverse { "suffix_sum" } else { "cumsum" },
                rank: a_layout.ndim(),
                max_rank: 2,
            },
        ));
    };
    let Some(output_layout) = rank2_layout(c_layout) else {
        return Err(crate::backend::WgpuBackendError::Validation(
            BackendError::UnsupportedRank {
                operation: if reverse { "suffix_sum" } else { "cumsum" },
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
    let result = if reverse {
        hephaestus_wgpu::suffix_sum_into(
            device,
            input,
            axis,
            output,
            hephaestus_core::BlockWidth::DEFAULT,
        )
    } else {
        hephaestus_wgpu::cumsum_into(
            device,
            input,
            axis,
            output,
            hephaestus_core::BlockWidth::DEFAULT,
        )
    };
    result.map_err(|source| crate::backend::WgpuBackendError::dispatch("scan", source))
}

impl<
    T: WgpuScalar
        + leto_ops::Scalar
        + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>
        + OpIdentity<hephaestus_wgpu::CumSumOp>
        + IdentityToken<hephaestus_wgpu::CumSumOp, hephaestus_wgpu::Wgsl>,
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
        reduction::dispatch_reduce(op, a, a_layout, axis, c, c_layout)
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
