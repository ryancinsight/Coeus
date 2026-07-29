use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use crate::CudaBackendError;
use coeus_core::{Layout, Storage};
use std::sync::Arc;

fn activation_tail_operation(op: coeus_ops::UnaryOp) -> Option<&'static str> {
    match op {
        coeus_ops::UnaryOp::Mish => Some("Mish"),
        coeus_ops::UnaryOp::MishGrad => Some("MishGrad"),
        coeus_ops::UnaryOp::Elu => Some("Elu"),
        coeus_ops::UnaryOp::EluGrad => Some("EluGrad"),
        _ => None,
    }
}

fn try_hephaestus_contiguous_binary<T>(
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

fn try_hephaestus_contiguous_unary<T>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
) -> Result<bool, CudaBackendError>
where
    T: CudaScalar + hephaestus_cuda::DialectScalar<hephaestus_cuda::CudaC>,
{
    let device = crate::backend::get_cuda_device();
    if activation_tail_operation(op).is_some() {
        let run_into = |result: hephaestus_cuda::Result<()>| {
            result
                .map(|_| true)
                .map_err(|source| CudaBackendError::dispatch("elementwise unary", source))
        };
        return match op {
            coeus_ops::UnaryOp::Mish => run_into(hephaestus_cuda::unary_elementwise_into::<
                hephaestus_cuda::MishOp,
                T,
            >(
                device,
                a.buffer.as_ref(),
                c.buffer.as_ref(),
                hephaestus_cuda::BlockWidth::DEFAULT,
            )),
            coeus_ops::UnaryOp::MishGrad => run_into(hephaestus_cuda::unary_elementwise_into::<
                hephaestus_cuda::MishGradOp,
                T,
            >(
                device,
                a.buffer.as_ref(),
                c.buffer.as_ref(),
                hephaestus_cuda::BlockWidth::DEFAULT,
            )),
            coeus_ops::UnaryOp::Elu => run_into(hephaestus_cuda::unary_elementwise_into::<
                hephaestus_cuda::EluOp,
                T,
            >(
                device,
                a.buffer.as_ref(),
                c.buffer.as_ref(),
                hephaestus_cuda::BlockWidth::DEFAULT,
            )),
            coeus_ops::UnaryOp::EluGrad => run_into(hephaestus_cuda::unary_elementwise_into::<
                hephaestus_cuda::EluGradOp,
                T,
            >(
                device,
                a.buffer.as_ref(),
                c.buffer.as_ref(),
                hephaestus_cuda::BlockWidth::DEFAULT,
            )),
            _ => Ok(false),
        };
    }
    if Arc::ptr_eq(&a.buffer, &c.buffer) {
        return Ok(false);
    }
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

#[allow(clippy::too_many_arguments)]
fn try_hephaestus_strided_binary<T>(
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
    if !can_route_dynamic_strided(&[a_layout], c_layout) {
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
        coeus_ops::UnaryOp::Mish => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::MishOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::MishGrad => run(hephaestus_cuda::unary_elementwise_strided_dyn_into::<
            hephaestus_cuda::MishGradOp,
            T,
        >(
            device,
            hephaestus_operand(a, a_layout),
            hephaestus_operand(c, c_layout),
            hephaestus_cuda::BlockWidth::DEFAULT,
        )),
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
        _ => Ok(false),
    }
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
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
        if get_cuda_context().is_some() {
            let Some(n) = kernels::checked_numel(c_layout) else {
                self.fallback_binary(op, a, a_layout, b, b_layout, c, c_layout)?;
                return Ok(());
            };
            // The contiguous kernel computes `c[i] = a[i] op b[i]` with no
            // broadcasting, so it is only valid when both operands already share
            // the output shape. A broadcast operand (e.g. `[3,1]` against
            // `[3,2]`) must go through the strided kernel, which resolves each
            // output coordinate against per-operand strides.
            let same_shape =
                a_layout.shape() == c_layout.shape() && b_layout.shape() == c_layout.shape();
            if same_shape
                && a_layout.is_contiguous()
                && b_layout.is_contiguous()
                && c_layout.is_contiguous()
            {
                if try_hephaestus_contiguous_binary(op, a, b, c)? {
                    return Ok(());
                }
                if kernels::launch_contiguous_binary(op, a, b, c, n) {
                    return Ok(());
                }
            } else if try_hephaestus_strided_binary(op, a, a_layout, b, b_layout, c, c_layout)?
                || kernels::launch_strided_binary(op, a, a_layout, b, b_layout, c, c_layout, n)
            {
                return Ok(());
            }
        }
        self.fallback_binary(op, a, a_layout, b, b_layout, c, c_layout)?;
        Ok(())
    }

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
        if let Some(operation) = activation_tail_operation(op) {
            if get_cuda_context().is_none() {
                return Err(CudaBackendError::ProviderUnavailable {
                    operation,
                    reason: "no CUDA context is available",
                });
            }
            let contiguous = a.len() == c.len()
                && a_layout.shape() == c_layout.shape()
                && a_layout.is_contiguous()
                && a_layout.offset() == 0
                && c_layout.is_contiguous()
                && c_layout.offset() == 0;
            if contiguous {
                if try_hephaestus_contiguous_unary(op, a, c)? {
                    return Ok(());
                }
            } else if can_route_dynamic_strided(&[a_layout], c_layout) {
                if try_hephaestus_strided_unary(op, a, a_layout, c, c_layout)? {
                    return Ok(());
                }
            } else {
                return Err(CudaBackendError::InvalidLayout {
                    operation,
                    reason: "layout is outside the provider-owned contiguous or strided contract",
                });
            }
            return Err(CudaBackendError::InvalidLayout {
                operation,
                reason: "provider-owned activation dispatch was not selected",
            });
        }
        if get_cuda_context().is_some() {
            let Some(n) = kernels::checked_numel(c_layout) else {
                self.fallback_unary(op, a, a_layout, c, c_layout)?;
                return Ok(());
            };
            if a_layout.is_contiguous() && c_layout.is_contiguous() {
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
        self.fallback_unary(op, a, a_layout, c, c_layout)?;
        Ok(())
    }

    pub(crate) fn cuda_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let a_f32 = cast_storage::<T, f32>(a);
            let b_f32 = cast_storage::<T, f32>(b);
            let mut c_f32 = cast_storage_mut::<T, f32>(c);
            if kernels::launch_matmul_tiled(
                &a_f32, &b_f32, &mut c_f32, a_layout, b_layout, c_layout,
            ) {
                return Ok(());
            }
        }
        self.fallback_matmul(a, a_layout, b, b_layout, c, c_layout)
    }

    pub(crate) fn cuda_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) -> Result<(), CudaBackendError> {
        if get_cuda_context().is_some()
            && kernels::dispatch_reduce(op, a, a_layout, axis, c, c_layout)
        {
            return Ok(());
        }
        self.fallback_reduce(op, a, a_layout, axis, c, c_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::activation_tail_operation;

    #[test]
    fn classifies_only_provider_owned_activation_tail_operations() {
        assert_eq!(
            activation_tail_operation(coeus_ops::UnaryOp::Mish),
            Some("Mish")
        );
        assert_eq!(
            activation_tail_operation(coeus_ops::UnaryOp::MishGrad),
            Some("MishGrad")
        );
        assert_eq!(
            activation_tail_operation(coeus_ops::UnaryOp::Elu),
            Some("Elu")
        );
        assert_eq!(
            activation_tail_operation(coeus_ops::UnaryOp::EluGrad),
            Some("EluGrad")
        );
        assert_eq!(activation_tail_operation(coeus_ops::UnaryOp::Silu), None);
    }
}
