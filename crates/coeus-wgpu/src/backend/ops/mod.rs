use crate::backend::{WgpuBackend, WgpuBackendError, WgpuScalar};
use crate::kernels;
use coeus_core::{Layout, Storage};
use hephaestus_core::BlockWidth;
use hephaestus_wgpu::{
    binary_elementwise_strided_into, unary_elementwise_strided_into, StridedOperand,
    MAX_STRIDED_RANK,
};
use leto::Layout as LetoLayout;
use std::sync::Arc;

mod attention;
mod conv;
mod impls;
mod matmul;
mod optim;
mod pool;
mod reduction;

// ── WGPU Hephaestus strided routing helpers ───────────────────────────────────

/// Guard: all layouts have rank ≤ MAX_STRIDED_RANK and the output has no
/// broadcast (zero-stride) dimensions where dim > 1.
fn can_route_strided_wgpu(layouts: &[&Layout], out: &Layout) -> bool {
    let max_rank = MAX_STRIDED_RANK;
    layouts
        .iter()
        .chain(std::iter::once(&out))
        .all(|l| l.ndim() <= max_rank)
        && !out
            .shape()
            .iter()
            .zip(out.strides())
            .any(|(&dim, &stride)| dim > 1 && stride == 0)
}

/// Convert a dynamic Coeus Layout to a `leto::Layout<N>`.
/// Pads a shorter layout on the left with size-1/stride-0 dimensions
/// so it can be broadcast against the target rank N.
macro_rules! coeus_to_leto_layout {
    ($layout:expr, $n:expr) => {{
        let rank = $layout.ndim();
        let pad = $n - rank.min($n);
        let shape: [usize; $n] = {
            let s = $layout.shape();
            let mut arr = [1usize; $n];
            for i in 0..rank.min($n) {
                arr[pad + i] = s[i];
            }
            arr
        };
        let strides: [isize; $n] = {
            let st = $layout.strides();
            let mut arr = [0isize; $n];
            for i in 0..rank.min($n) {
                arr[pad + i] = st[i] as isize;
            }
            arr
        };
        LetoLayout::new(shape, strides, $layout.offset())
    }};
}

/// Dispatch a binary Hephaestus strided op at the rank determined by `out.ndim()`.
/// Returns `Ok(true)` when dispatched, `Ok(false)` when unsupported.
fn try_hephaestus_strided_binary_wgpu<
    T: WgpuScalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>,
>(
    op: coeus_ops::BinaryOp,
    a_buf: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    b_buf: &crate::backend::WgpuStorage<T>,
    b_layout: &Layout,
    c_buf: &crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<bool, WgpuBackendError> {
    use coeus_ops::BinaryOp;

    macro_rules! dispatch_n {
        ($n:expr) => {{
            let la = coeus_to_leto_layout!(a_layout, $n);
            let lb = coeus_to_leto_layout!(b_layout, $n);
            let lc = coeus_to_leto_layout!(c_layout, $n);
            let a_op = StridedOperand {
                buffer: a_buf.buffer.as_ref(),
                layout: &la,
            };
            let b_op = StridedOperand {
                buffer: b_buf.buffer.as_ref(),
                layout: &lb,
            };
            let c_op = StridedOperand {
                buffer: c_buf.buffer.as_ref(),
                layout: &lc,
            };
            let ok = |r: hephaestus_wgpu::Result<()>| {
                r.map(|_| true)
                    .map_err(|source| WgpuBackendError::dispatch("elementwise binary", source))
            };
            let dev = &crate::backend::get_wgpu_context().hephaestus_device;
            match op {
                BinaryOp::Add => ok(binary_elementwise_strided_into::<
                    hephaestus_wgpu::AddOp,
                    T,
                    $n,
                >(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Sub => ok(binary_elementwise_strided_into::<
                    hephaestus_wgpu::SubOp,
                    T,
                    $n,
                >(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Mul => ok(binary_elementwise_strided_into::<
                    hephaestus_wgpu::MulOp,
                    T,
                    $n,
                >(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Div => ok(binary_elementwise_strided_into::<
                    hephaestus_wgpu::DivOp,
                    T,
                    $n,
                >(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                _ => Ok(false),
            }
        }};
    }

    match c_layout.ndim().max(a_layout.ndim()).max(b_layout.ndim()) {
        1 => dispatch_n!(1),
        2 => dispatch_n!(2),
        3 => dispatch_n!(3),
        4 => dispatch_n!(4),
        _ => Ok(false),
    }
}

/// Dispatch a unary Hephaestus strided op at the rank determined by `out.ndim()`.
fn try_hephaestus_strided_unary_wgpu<
    T: WgpuScalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>,
>(
    op: coeus_ops::UnaryOp,
    a_buf: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    c_buf: &crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> Result<bool, WgpuBackendError> {
    use coeus_ops::UnaryOp;

    macro_rules! dispatch_n {
        ($n:expr) => {{
            let la = coeus_to_leto_layout!(a_layout, $n);
            let lc = coeus_to_leto_layout!(c_layout, $n);
            let a_op = StridedOperand {
                buffer: a_buf.buffer.as_ref(),
                layout: &la,
            };
            let c_op = StridedOperand {
                buffer: c_buf.buffer.as_ref(),
                layout: &lc,
            };
            let ok = |r: hephaestus_wgpu::Result<()>| {
                r.map(|_| true)
                    .map_err(|source| WgpuBackendError::dispatch("elementwise unary", source))
            };
            let dev = &crate::backend::get_wgpu_context().hephaestus_device;
            match op {
                UnaryOp::Sin => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::SinOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Cos => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::CosOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Exp => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::ExpOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Log => ok(
                    unary_elementwise_strided_into::<hephaestus_wgpu::LnOp, T, $n>(
                        dev,
                        a_op,
                        c_op,
                        BlockWidth::DEFAULT,
                    ),
                ),
                UnaryOp::Neg => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::NegOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Abs => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::AbsOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Sqrt => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::SqrtOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Recip => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::RecipOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Lgamma => ok(unary_elementwise_strided_into::<
                    hephaestus_wgpu::LgammaOp,
                    T,
                    $n,
                >(dev, a_op, c_op, BlockWidth::DEFAULT)),
                _ => Ok(false),
            }
        }};
    }

    match c_layout.ndim().max(a_layout.ndim()) {
        1 => dispatch_n!(1),
        2 => dispatch_n!(2),
        3 => dispatch_n!(3),
        4 => dispatch_n!(4),
        _ => Ok(false),
    }
}

fn try_hephaestus_contiguous_binary<
    T: WgpuScalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>,
>(
    op: coeus_ops::BinaryOp,
    a: &crate::storage::WgpuStorage<T>,
    b: &crate::storage::WgpuStorage<T>,
    c: &mut crate::storage::WgpuStorage<T>,
) -> Result<bool, WgpuBackendError> {
    if Arc::ptr_eq(&a.buffer, &c.buffer) || Arc::ptr_eq(&b.buffer, &c.buffer) {
        return Ok(false);
    }
    let ctx = crate::backend::get_wgpu_context();
    let run = |result: hephaestus_wgpu::Result<()>| {
        result
            .map(|_| true)
            .map_err(|source| WgpuBackendError::dispatch("elementwise binary", source))
    };
    match op {
        coeus_ops::BinaryOp::Add => run(hephaestus_wgpu::binary_elementwise_into::<
            hephaestus_wgpu::AddOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            b.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Sub => run(hephaestus_wgpu::binary_elementwise_into::<
            hephaestus_wgpu::SubOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            b.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Mul => run(hephaestus_wgpu::binary_elementwise_into::<
            hephaestus_wgpu::MulOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            b.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::BinaryOp::Div => run(hephaestus_wgpu::binary_elementwise_into::<
            hephaestus_wgpu::DivOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            b.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        _ => Ok(false),
    }
}

fn try_hephaestus_contiguous_unary<
    T: WgpuScalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>,
>(
    op: coeus_ops::UnaryOp,
    a: &crate::storage::WgpuStorage<T>,
    c: &mut crate::storage::WgpuStorage<T>,
) -> Result<bool, WgpuBackendError> {
    if Arc::ptr_eq(&a.buffer, &c.buffer) {
        return Ok(false);
    }
    let ctx = crate::backend::get_wgpu_context();
    let run = |result: hephaestus_wgpu::Result<()>| {
        result
            .map(|_| true)
            .map_err(|source| WgpuBackendError::dispatch("elementwise unary", source))
    };
    match op {
        coeus_ops::UnaryOp::Sin => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::SinOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Cos => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::CosOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Exp => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::ExpOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Log => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::LnOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Neg => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::NegOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Abs => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::AbsOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Sqrt => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::SqrtOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Recip => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::RecipOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        coeus_ops::UnaryOp::Lgamma => run(hephaestus_wgpu::unary_elementwise_into::<
            hephaestus_wgpu::LgammaOp,
            T,
        >(
            &ctx.hephaestus_device,
            a.buffer.as_ref(),
            c.buffer.as_ref(),
            BlockWidth::DEFAULT,
        )),
        _ => Ok(false),
    }
}

impl<T: WgpuScalar + leto_ops::Scalar + hephaestus_wgpu::DialectScalar<hephaestus_wgpu::Wgsl>>
    coeus_ops::ElementwiseOps<T> for WgpuBackend
{
    #[inline]
    fn elementwise_binary(
        &self,
        op: coeus_ops::BinaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), WgpuBackendError> {
        WgpuBackendError::validate_layout(a_layout)?;
        WgpuBackendError::validate_layout(b_layout)?;
        WgpuBackendError::validate_layout(c_layout)?;
        if a.len() == c.len()
            && b.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && b_layout.is_contiguous()
            && b_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            if !try_hephaestus_contiguous_binary(op, a, b, c)? {
                kernels::dispatch_contiguous_binary::<T>(
                    op,
                    &a.buffer,
                    &b.buffer,
                    &c.buffer,
                    c.len(),
                )?;
            }
        } else if can_route_strided_wgpu(&[a_layout, b_layout], c_layout)
            && try_hephaestus_strided_binary_wgpu(op, a, a_layout, b, b_layout, c, c_layout)?
        {
        } else {
            kernels::dispatch_binary::<T>(
                op,
                &a.buffer,
                a_layout,
                &b.buffer,
                b_layout,
                &c.buffer,
                c_layout,
                c.len(),
            )?;
        }
        Ok(())
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: coeus_ops::UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) -> Result<(), WgpuBackendError> {
        WgpuBackendError::validate_layout(a_layout)?;
        WgpuBackendError::validate_layout(c_layout)?;
        if a.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            if !try_hephaestus_contiguous_unary(op, a, c)? {
                kernels::dispatch_contiguous_unary::<T>(op, &a.buffer, &c.buffer, c.len())?;
            }
        } else if can_route_strided_wgpu(&[a_layout], c_layout)
            && try_hephaestus_strided_unary_wgpu(op, a, a_layout, c, c_layout)?
        {
        } else {
            kernels::dispatch_unary::<T>(op, &a.buffer, a_layout, &c.buffer, c_layout, c.len())?;
        }
        Ok(())
    }
}
