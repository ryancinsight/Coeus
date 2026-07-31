use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::{Layout, Storage};
use std::sync::Arc;

mod binary;
mod validation;

use validation::validate_unary_layouts;

fn try_hephaestus_contiguous_unary<T>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
) -> Result<bool, CudaBackendError>
where
    T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    if Arc::ptr_eq(&a.buffer, &c.buffer) {
        return Ok(false);
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<hephaestus_cuda::CudaBuffer<T>>,
               c: &mut CudaStorage<T>| {
        let buffer =
            result.map_err(|source| CudaBackendError::dispatch("elementwise unary", source))?;
        c.buffer = Arc::new(buffer);
        Ok(true)
    };
    match op {
        coeus_ops::UnaryOp::Sin => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::SinOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Cos => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::CosOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Exp => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::ExpOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Log => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::LnOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Neg => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::NegOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Abs => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::AbsOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Sqrt => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::SqrtOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Recip => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::RecipOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Gelu => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::GeluOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::GeluGrad => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::GeluGradOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::GeluTanh => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::GeluTanhOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::GeluTanhGrad => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::GeluTanhGradOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Elu => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::EluOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::EluGrad => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::EluGradOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::Softplus => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::SoftplusOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::UnaryOp::SoftplusGrad => run(
            hephaestus_cuda::unary_elementwise::<hephaestus_cuda::SoftplusGradOp, T>(
                device,
                a.buffer.as_ref(),
            ),
            c,
        ),
        _ => Ok(false),
    }
}

#[inline]
fn hephaestus_operand<'a, T>(
    storage: &'a CudaStorage<T>,
    layout: &'a Layout,
) -> hephaestus_cuda::StridedOperandDyn<'a, T> {
    hephaestus_cuda::StridedOperandDyn {
        buffer: storage.buffer.as_ref(),
        layout: hephaestus_cuda::StridedLayout {
            shape: layout.shape(),
            strides: layout.strides(),
            offset: layout.offset(),
        },
    }
}

#[inline]
fn can_route_dynamic_strided(layouts: &[&Layout], out: &Layout) -> bool {
    layouts
        .iter()
        .chain(std::iter::once(&out))
        .all(|layout| layout.ndim() <= hephaestus_cuda::MAX_STRIDED_RANK)
        && !out
            .shape()
            .iter()
            .zip(out.strides())
            .any(|(&dim, &stride)| dim > 1 && stride == 0)
}

fn try_hephaestus_strided_unary<T>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
) -> Result<bool, CudaBackendError>
where
    T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    if Arc::ptr_eq(&a.buffer, &c.buffer) || !can_route_dynamic_strided(&[a_layout], c_layout) {
        return Ok(false);
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<()>| {
        result
            .map(|_| true)
            .map_err(|source| CudaBackendError::dispatch("elementwise unary", source))
    };
    match op {
        coeus_ops::UnaryOp::Sin => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::SinOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Cos => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::CosOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Exp => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::ExpOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Log => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::LnOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Neg => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::NegOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Abs => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::AbsOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Sqrt => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::SqrtOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Recip => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::RecipOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Gelu => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::GeluOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::GeluGrad => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::GeluGradOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::GeluTanh => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::GeluTanhOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::GeluTanhGrad => {
            run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
                hephaestus_cuda::GeluTanhGradOp,
                T,
            >(
                device,
                hephaestus_operand(a, a_layout),
                hephaestus_operand(c, c_layout),
                hephaestus_cuda::BlockWidth::DEFAULT,
            ))
        }
        coeus_ops::UnaryOp::Elu => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::EluOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::EluGrad => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::EluGradOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Softplus => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::SoftplusOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::SoftplusGrad => {
            run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
                hephaestus_cuda::SoftplusGradOp,
                T,
            >(
                device,
                hephaestus_operand(a, a_layout),
                hephaestus_operand(c, c_layout),
                hephaestus_cuda::BlockWidth::DEFAULT,
            ))
        }
        _ => Ok(false),
    }
}

impl CudaBackend {
    pub(crate) fn cuda_elementwise_unary<T>(
        &self,
        op: coeus_ops::UnaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), CudaBackendError>
    where
        T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
    {
        validate_unary_layouts(
            a_layout,
            a.len(),
            c_layout,
            c.len(),
            Arc::ptr_eq(&a.buffer, &c.buffer),
        )?;
        let n = kernels::checked_numel(c_layout).ok_or_else(|| {
            CudaBackendError::kernel(
                "elementwise unary",
                "output element count exceeds the CUDA dispatch ABI",
            )
        })?;
        if n == 0 {
            return Ok(());
        }
        if get_cuda_context().is_some() {
            if a_layout.is_contiguous()
                && a_layout.offset() == 0
                && c_layout.is_contiguous()
                && c_layout.offset() == 0
            {
                if try_hephaestus_contiguous_unary(op, a, c)? {
                    return Ok(());
                }
                if kernels::launch_contiguous_unary(op, a, c, n) {
                    return Ok(());
                }
            } else {
                if try_hephaestus_strided_unary(op, a, a_layout, c, c_layout)?
                    || kernels::launch_strided_unary(op, a, a_layout, c, c_layout, n)
                {
                    return Ok(());
                }
            }
        }
        Err(CudaBackendError::kernel(
            "elementwise unary",
            "native CUDA dispatch requirements are not satisfied",
        ))
    }
}
