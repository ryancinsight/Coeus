use super::cast::{cast_storage, cast_storage_mut};
use crate::backend::{CudaBackend, CudaScalar};
use crate::driver::get_cuda_context;
use crate::kernels;
use crate::storage::CudaStorage;
use coeus_core::Layout;
use std::sync::Arc;

fn try_hephaestus_contiguous_binary<T>(
    op: coeus_ops::BinaryOp,
    a: &CudaStorage<T>,
    b: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
) -> bool
where
    T: CudaScalar + hephaestus_cuda::CudaScalar,
{
    if Arc::ptr_eq(&a.buffer, &c.buffer) || Arc::ptr_eq(&b.buffer, &c.buffer) {
        return false;
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<hephaestus_cuda::CudaBuffer<T>>,
               c: &mut CudaStorage<T>| {
        c.buffer = Arc::new(result.expect("hephaestus-cuda contiguous binary dispatch failed"));
        true
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
    }
}

fn try_hephaestus_contiguous_unary<T>(
    op: coeus_ops::UnaryOp,
    a: &CudaStorage<T>,
    c: &mut CudaStorage<T>,
) -> bool
where
    T: CudaScalar + hephaestus_cuda::CudaScalar,
{
    if Arc::ptr_eq(&a.buffer, &c.buffer) {
        return false;
    }
    let device = crate::backend::get_cuda_device();
    let run = |result: hephaestus_cuda::Result<hephaestus_cuda::CudaBuffer<T>>,
               c: &mut CudaStorage<T>| {
        c.buffer = Arc::new(result.expect("hephaestus-cuda contiguous unary dispatch failed"));
        true
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
        _ => false,
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
    ) where
        T: CudaScalar + hephaestus_cuda::CudaScalar,
    {
        if get_cuda_context().is_some() {
            let n = c_layout.shape().iter().product();
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
                if try_hephaestus_contiguous_binary(op, a, b, c) {
                    return;
                }
                if kernels::launch_contiguous_binary(op, a, b, c, n) {
                    return;
                }
            } else if kernels::launch_strided_binary(op, a, a_layout, b, b_layout, c, c_layout, n) {
                return;
            }
        }
        self.fallback_binary(op, a, a_layout, b, b_layout, c, c_layout);
    }

    pub(crate) fn cuda_elementwise_unary<T>(
        &self,
        op: coeus_ops::UnaryOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) where
        T: CudaScalar + hephaestus_cuda::CudaScalar,
    {
        if get_cuda_context().is_some() {
            let n = c_layout.shape().iter().product();
            if a_layout.is_contiguous() && c_layout.is_contiguous() {
                if try_hephaestus_contiguous_unary(op, a, c) {
                    return;
                }
                if kernels::launch_contiguous_unary(op, a, c, n) {
                    return;
                }
            } else {
                if kernels::launch_strided_unary(op, a, a_layout, c, c_layout, n) {
                    return;
                }
            }
        }
        self.fallback_unary(op, a, a_layout, c, c_layout);
    }

    pub(crate) fn cuda_matmul<T: CudaScalar>(
        &self,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        b: &CudaStorage<T>,
        b_layout: &Layout,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            let a_f32 = cast_storage::<T, f32>(a);
            let b_f32 = cast_storage::<T, f32>(b);
            let mut c_f32 = cast_storage_mut::<T, f32>(c);
            if kernels::launch_matmul_tiled(
                &a_f32, &b_f32, &mut c_f32, a_layout, b_layout, c_layout,
            ) {
                return;
            }
        }
        self.fallback_matmul(a, a_layout, b, b_layout, c, c_layout);
    }

    pub(crate) fn cuda_reduce<T: CudaScalar>(
        &self,
        op: coeus_ops::ReductionOp,
        a: &CudaStorage<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut CudaStorage<T>,
        c_layout: &Layout,
    ) {
        if get_cuda_context().is_some()
            && kernels::dispatch_reduce(op, a, a_layout, axis, c, c_layout)
        {
            return;
        }
        self.fallback_reduce(op, a, a_layout, axis, c, c_layout);
    }
}
