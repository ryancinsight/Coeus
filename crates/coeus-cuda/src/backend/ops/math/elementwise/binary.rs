use super::{can_route_dynamic_strided, hephaestus_operand};
use crate::CudaBackendError;
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::{Layout, Storage};
use std::sync::Arc;

use super::validation::validate_binary_layouts;

fn try_hephaestus_contiguous<T>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
) -> Result<bool, CudaBackendError>
where
    T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    if Arc::ptr_eq(&a.buffer, &c.buffer) || Arc::ptr_eq(&b.buffer, &c.buffer) {
        return Ok(false);
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<hephaestus_cuda::CudaBuffer<T>>,
               c: &mut CudaStorage<T>| {
        let buffer =
            result.map_err(|source| CudaBackendError::dispatch("elementwise binary", source))?;
        c.buffer = Arc::new(buffer);
        Ok(true)
    };
    match op {
        coeus_ops::BinaryOp::Add => run(
            hephaestus_cuda::binary_elementwise::<hephaestus_cuda::AddOp, T>(
                device,
                a.buffer.as_ref(),
                b.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::BinaryOp::Sub => run(
            hephaestus_cuda::binary_elementwise::<hephaestus_cuda::SubOp, T>(
                device,
                a.buffer.as_ref(),
                b.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::BinaryOp::Mul => run(
            hephaestus_cuda::binary_elementwise::<hephaestus_cuda::MulOp, T>(
                device,
                a.buffer.as_ref(),
                b.buffer.as_ref(),
            ),
            c,
        ),
        coeus_ops::BinaryOp::Div => run(
            hephaestus_cuda::binary_elementwise::<hephaestus_cuda::DivOp, T>(
                device,
                a.buffer.as_ref(),
                b.buffer.as_ref(),
            ),
            c,
        ),
        _ => Ok(false),
    }
}

fn try_hephaestus_strided<T>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    a_layout: &Layout,
    b: &CudaStorage<T>,
    b_layout: &Layout,
    c: &mut CudaStorage<T>,
    c_layout: &Layout,
) -> Result<bool, CudaBackendError>
where
    T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    if !can_route_dynamic_strided(&[a_layout, b_layout], c_layout) {
        return Ok(false);
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<()>| {
        result
            .map(|_| true)
            .map_err(|source| CudaBackendError::dispatch("elementwise binary", source))
    };
    match op {
        coeus_ops::BinaryOp::Add => run(hephaestus_cuda::binary_elementwise_strided_dyn_into::<
            hephaestus_cuda::AddOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(b, b_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Sub => run(hephaestus_cuda::binary_elementwise_strided_dyn_into::<
            hephaestus_cuda::SubOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(b, b_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Mul => run(hephaestus_cuda::binary_elementwise_strided_dyn_into::<
            hephaestus_cuda::MulOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(b, b_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Div => run(hephaestus_cuda::binary_elementwise_strided_dyn_into::<
            hephaestus_cuda::DivOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(b, b_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        _ => Ok(false),
    }
}

impl CudaBackend {
    #[expect(
        clippy::too_many_arguments,
        reason = "backend dispatch receives three explicit buffer-layout pairs"
    )]
    pub(crate) fn cuda_elementwise_binary<T>(
        &self,
        op: coeus_ops::BinaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), CudaBackendError>
    where
        T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
    {
        validate_binary_layouts(
            a_layout,
            a.len(),
            b_layout,
            b.len(),
            c_layout,
            c.len(),
            Arc::ptr_eq(&a.buffer, &c.buffer),
            Arc::ptr_eq(&b.buffer, &c.buffer),
        )?;
        let n = kernels::checked_numel(c_layout).ok_or_else(|| {
            CudaBackendError::kernel(
                "elementwise binary",
                "output element count exceeds the CUDA dispatch ABI",
            )
        })?;
        if n == 0 {
            return Ok(());
        }
        if get_cuda_context().is_some() {
            let same_shape =
                a_layout.shape() == c_layout.shape() && b_layout.shape() == c_layout.shape();
            if same_shape
                && a_layout.is_contiguous()
                && a_layout.offset() == 0
                && b_layout.is_contiguous()
                && b_layout.offset() == 0
                && c_layout.is_contiguous()
                && c_layout.offset() == 0
            {
                if try_hephaestus_contiguous(op, a, b, c)?
                    || kernels::launch_contiguous_binary(op, a, b, c, n)
                {
                    return Ok(());
                }
            } else if try_hephaestus_strided(op, a, a_layout, b, b_layout, c, c_layout)?
                || kernels::launch_strided_binary(op, a, a_layout, b, b_layout, c, c_layout, n)
            {
                return Ok(());
            }
        }
        Err(CudaBackendError::kernel(
            "elementwise binary",
            "native CUDA dispatch requirements are not satisfied",
        ))
    }
}
