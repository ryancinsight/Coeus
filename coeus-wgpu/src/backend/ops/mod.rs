use crate::backend::{WgpuBackend, WgpuScalar};
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
/// Returns `true` when dispatched, `false` when the rank falls outside [1, 4].
fn try_hephaestus_strided_binary_wgpu<T: WgpuScalar + hephaestus_wgpu::WgslScalar>(
    op: coeus_ops::BinaryOp,
    a_buf: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    b_buf: &crate::backend::WgpuStorage<T>,
    b_layout: &Layout,
    c_buf: &crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> bool {
    use coeus_ops::BinaryOp;

    macro_rules! dispatch_n {
        ($n:expr) => {{
            let la = coeus_to_leto_layout!(a_layout, $n);
            let lb = coeus_to_leto_layout!(b_layout, $n);
            let lc = coeus_to_leto_layout!(c_layout, $n);
            let a_op = StridedOperand { buffer: a_buf.buffer.as_ref(), layout: &la };
            let b_op = StridedOperand { buffer: b_buf.buffer.as_ref(), layout: &lb };
            let c_op = StridedOperand { buffer: c_buf.buffer.as_ref(), layout: &lc };
            let ok = |r: hephaestus_wgpu::Result<()>| {
                r.expect("hephaestus-wgpu strided binary dispatch failed");
                true
            };
            let dev = &crate::backend::get_wgpu_context().hephaestus_device;
            match op {
                BinaryOp::Add => ok(binary_elementwise_strided_into::<hephaestus_wgpu::AddOp, T, $n>(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Sub => ok(binary_elementwise_strided_into::<hephaestus_wgpu::SubOp, T, $n>(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Mul => ok(binary_elementwise_strided_into::<hephaestus_wgpu::MulOp, T, $n>(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
                BinaryOp::Div => ok(binary_elementwise_strided_into::<hephaestus_wgpu::DivOp, T, $n>(dev, a_op, b_op, c_op, BlockWidth::DEFAULT)),
            }
        }};
    }

    match c_layout.ndim().max(a_layout.ndim()).max(b_layout.ndim()) {
        1 => dispatch_n!(1),
        2 => dispatch_n!(2),
        3 => dispatch_n!(3),
        4 => dispatch_n!(4),
        _ => false,
    }
}

/// Dispatch a unary Hephaestus strided op at the rank determined by `out.ndim()`.
fn try_hephaestus_strided_unary_wgpu<T: WgpuScalar + hephaestus_wgpu::WgslScalar>(
    op: coeus_ops::UnaryOp,
    a_buf: &crate::backend::WgpuStorage<T>,
    a_layout: &Layout,
    c_buf: &crate::backend::WgpuStorage<T>,
    c_layout: &Layout,
) -> bool {
    use coeus_ops::UnaryOp;

    macro_rules! dispatch_n {
        ($n:expr) => {{
            let la = coeus_to_leto_layout!(a_layout, $n);
            let lc = coeus_to_leto_layout!(c_layout, $n);
            let a_op = StridedOperand { buffer: a_buf.buffer.as_ref(), layout: &la };
            let c_op = StridedOperand { buffer: c_buf.buffer.as_ref(), layout: &lc };
            let ok = |r: hephaestus_wgpu::Result<()>| {
                r.expect("hephaestus-wgpu strided unary dispatch failed");
                true
            };
            let dev = &crate::backend::get_wgpu_context().hephaestus_device;
            match op {
                UnaryOp::Sin => ok(unary_elementwise_strided_into::<hephaestus_wgpu::SinOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Cos => ok(unary_elementwise_strided_into::<hephaestus_wgpu::CosOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Exp => ok(unary_elementwise_strided_into::<hephaestus_wgpu::ExpOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Log => ok(unary_elementwise_strided_into::<hephaestus_wgpu::LnOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Neg => ok(unary_elementwise_strided_into::<hephaestus_wgpu::NegOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Abs => ok(unary_elementwise_strided_into::<hephaestus_wgpu::AbsOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Sqrt => ok(unary_elementwise_strided_into::<hephaestus_wgpu::SqrtOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                UnaryOp::Recip => ok(unary_elementwise_strided_into::<hephaestus_wgpu::RecipOp, T, $n>(dev, a_op, c_op, BlockWidth::DEFAULT)),
                _ => false,
            }
        }};
    }

    match c_layout.ndim().max(a_layout.ndim()) {
        1 => dispatch_n!(1),
        2 => dispatch_n!(2),
        3 => dispatch_n!(3),
        4 => dispatch_n!(4),
        _ => false,
    }
}

fn try_hephaestus_contiguous_binary<T: WgpuScalar + hephaestus_wgpu::WgslScalar>(
    op: coeus_ops::BinaryOp,
    a: &crate::storage::WgpuStorage<T>,
    b: &crate::storage::WgpuStorage<T>,
    c: &mut crate::storage::WgpuStorage<T>,
) -> bool {
    if Arc::ptr_eq(&a.buffer, &c.buffer) || Arc::ptr_eq(&b.buffer, &c.buffer) {
        return false;
    }
    let ctx = crate::backend::get_wgpu_context();
    let run = |result: hephaestus_wgpu::Result<()>| {
        result.expect("hephaestus-wgpu contiguous binary dispatch failed");
        true
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
    }
}

fn try_hephaestus_contiguous_unary<T: WgpuScalar + hephaestus_wgpu::WgslScalar>(
    op: coeus_ops::UnaryOp,
    a: &crate::storage::WgpuStorage<T>,
    c: &mut crate::storage::WgpuStorage<T>,
) -> bool {
    if Arc::ptr_eq(&a.buffer, &c.buffer) {
        return false;
    }
    let ctx = crate::backend::get_wgpu_context();
    let run = |result: hephaestus_wgpu::Result<()>| {
        result.expect("hephaestus-wgpu contiguous unary dispatch failed");
        true
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
        _ => false,
    }
}

impl<T: WgpuScalar + leto_ops::Scalar + hephaestus_wgpu::WgslScalar> coeus_ops::BackendOps<T>
    for WgpuBackend
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
    ) {
        if a.len() == c.len()
            && b.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && b_layout.is_contiguous()
            && b_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            if !try_hephaestus_contiguous_binary(op, a, b, c) {
                kernels::dispatch_contiguous_binary::<T>(
                    op,
                    &a.buffer,
                    &b.buffer,
                    &c.buffer,
                    c.len(),
                );
            }
        } else if can_route_strided_wgpu(&[a_layout, b_layout], c_layout)
            && try_hephaestus_strided_binary_wgpu(op, a, a_layout, b, b_layout, c, c_layout)
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
            );
        }
    }

    #[inline]
    fn elementwise_unary(
        &self,
        op: coeus_ops::UnaryOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        if a.len() == c.len()
            && a_layout.is_contiguous()
            && a_layout.offset() == 0
            && c_layout.is_contiguous()
            && c_layout.offset() == 0
        {
            if !try_hephaestus_contiguous_unary(op, a, c) {
                kernels::dispatch_contiguous_unary::<T>(op, &a.buffer, &c.buffer, c.len());
            }
        } else if can_route_strided_wgpu(&[a_layout], c_layout)
            && try_hephaestus_strided_unary_wgpu(op, a, a_layout, c, c_layout)
        {
        } else {
            kernels::dispatch_unary::<T>(op, &a.buffer, a_layout, &c.buffer, c_layout, c.len());
        }
    }

    #[inline]
    fn matmul(
        &self,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        b: &Self::DeviceBuffer<T>,
        b_layout: &Layout,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        kernels::dispatch_matmul::<T>(
            &a.buffer, a_layout, &b.buffer, b_layout, &c.buffer, c_layout,
        );
    }

    #[inline]
    fn reduce(
        &self,
        op: coeus_ops::ReductionOp,
        a: &Self::DeviceBuffer<T>,
        a_layout: &Layout,
        axis: usize,
        c: &mut Self::DeviceBuffer<T>,
        c_layout: &Layout,
    ) {
        kernels::dispatch_reduce::<T>(op, &a.buffer, a_layout, axis, &c.buffer, c_layout);
    }

    #[inline]
    fn conv1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        let out_numel = output_layout.shape().iter().product::<usize>();
        kernels::dispatch_conv1d::<T>(kernels::Conv1dDispatch {
            input: &input.buffer,
            weight: &weight.buffer,
            bias: bias.map(|b| b.buffer.raw()),
            output: &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            out_numel,
        });
    }

    #[inline]
    fn conv1d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        kernels::dispatch_conv1d_backward::<T>(kernels::Conv1dBackwardDispatch {
            grad_out: &grad_out.buffer,
            grad_out_layout,
            input: &input.buffer,
            input_layout,
            weight: &weight.buffer,
            weight_layout,
            grad_input: grad_input.map(|gi| gi.buffer.raw()),
            grad_input_layout,
            grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
            grad_weight_layout,
            grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
            stride,
            padding,
            dilation,
        });
    }

    #[inline]
    fn conv2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        let out_numel = output_layout.shape().iter().product::<usize>();
        kernels::dispatch_conv2d::<T>(kernels::Conv2dDispatch {
            input: &input.buffer,
            weight: &weight.buffer,
            bias: bias.map(|b| b.buffer.raw()),
            output: &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            out_numel,
        });
    }

    #[inline]
    fn conv2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        kernels::dispatch_conv2d_backward::<T>(kernels::Conv2dBackwardDispatch {
            grad_out: &grad_out.buffer,
            grad_out_layout,
            input: &input.buffer,
            input_layout,
            weight: &weight.buffer,
            weight_layout,
            grad_input: grad_input.map(|gi| gi.buffer.raw()),
            grad_input_layout,
            grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
            grad_weight_layout,
            grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
            stride,
            padding,
            dilation,
        });
    }

    #[inline]
    fn conv3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        kernels::dispatch_conv3d::<T>(kernels::Conv3dDispatch {
            input: &input.buffer,
            weight: &weight.buffer,
            bias: bias.map(|b| b.buffer.raw()),
            output: &output.buffer,
            input_layout,
            weight_layout,
            output_layout,
            stride,
            padding,
            dilation,
            out_numel: output.len(),
        });
    }

    #[inline]
    fn conv3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        grad_input: Option<&mut Self::DeviceBuffer<T>>,
        grad_input_layout: &Layout,
        grad_weight: Option<&mut Self::DeviceBuffer<T>>,
        grad_weight_layout: &Layout,
        grad_bias: Option<&mut Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) {
        kernels::dispatch_conv3d_backward::<T>(kernels::Conv3dBackwardDispatch {
            grad_out: &grad_out.buffer,
            grad_out_layout,
            input: &input.buffer,
            input_layout,
            weight: &weight.buffer,
            weight_layout,
            grad_input: grad_input.map(|gi| gi.buffer.raw()),
            grad_input_layout,
            grad_weight: grad_weight.map(|gw| gw.buffer.raw()),
            grad_weight_layout,
            grad_bias: grad_bias.map(|gb| gb.buffer.raw()),
            stride,
            padding,
            dilation,
        });
    }

    #[inline]
    fn conv_transpose1d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        let _ = output_padding; // encoded in output_layout shape, not the kernel loop
                                // input [n, c_in, l] / weight [c_in, c_out, k] / output [n, c_out, l_out]
        kernels::dispatch_conv_transpose1d(kernels::ConvTranspose1dDispatch {
            input: input.buffer.raw(),
            weight: weight.buffer.raw(),
            bias: bias.map(|b| b.buffer.raw()),
            output: output.buffer.raw(),
            n: input_layout.shape()[0],
            c_in: input_layout.shape()[1],
            l: input_layout.shape()[2],
            c_out: weight_layout.shape()[1],
            k: weight_layout.shape()[2],
            l_out: output_layout.shape()[2],
            stride,
            padding,
            dilation,
        });
    }

    #[inline]
    fn conv_transpose2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        weight: &Self::DeviceBuffer<T>,
        weight_layout: &Layout,
        bias: Option<&Self::DeviceBuffer<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        let _ = output_padding; // encoded in output_layout shape, not the kernel loop
                                // input [n, c_in, h, w] / weight [c_in, c_out, kh, kw] / output [n, c_out, h_out, w_out]
        kernels::dispatch_conv_transpose2d(kernels::ConvTranspose2dDispatch {
            input: input.buffer.raw(),
            weight: weight.buffer.raw(),
            bias: bias.map(|b| b.buffer.raw()),
            output: output.buffer.raw(),
            n: input_layout.shape()[0],
            c_in: input_layout.shape()[1],
            h: input_layout.shape()[2],
            w: input_layout.shape()[3],
            c_out: weight_layout.shape()[1],
            kh: weight_layout.shape()[2],
            kw: weight_layout.shape()[3],
            h_out: output_layout.shape()[2],
            w_out: output_layout.shape()[3],
            stride,
            padding,
            dilation,
        });
    }

    #[inline]
    fn max_pool2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        kernels::dispatch_max_pool2d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

    #[inline]
    fn max_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) {
        kernels::dispatch_max_pool2d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

    #[inline]
    fn avg_pool2d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        kernels::dispatch_avg_pool2d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

    #[inline]
    fn avg_pool2d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) {
        kernels::dispatch_avg_pool2d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

    #[inline]
    fn max_pool3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        kernels::dispatch_max_pool3d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

    #[inline]
    fn max_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) {
        kernels::dispatch_max_pool3d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

    #[inline]
    fn avg_pool3d(
        &self,
        input: &Self::DeviceBuffer<T>,
        input_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
    ) {
        kernels::dispatch_avg_pool3d::<T>(
            &input.buffer,
            input_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &output.buffer,
            output_layout,
            output.len(),
        );
    }

    #[inline]
    fn avg_pool3d_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        grad_out_layout: &Layout,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        grad_input: &mut Self::DeviceBuffer<T>,
        grad_input_layout: &Layout,
    ) {
        kernels::dispatch_avg_pool3d_backward::<T>(
            &grad_out.buffer,
            grad_out_layout,
            kernel_size,
            stride,
            padding,
            dilation,
            &grad_input.buffer,
            grad_input_layout,
            grad_input.len(),
        );
    }

    #[inline]
    fn sdp_attention(
        &self,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        key_padding_mask: Option<&Self::DeviceBuffer<T>>,
        key_padding_mask_layout: Option<&Layout>,
        is_causal: bool,
        scale: T,
        output: &mut Self::DeviceBuffer<T>,
        output_layout: &Layout,
        attn_weights: &mut Self::DeviceBuffer<T>,
        attn_weights_layout: &Layout,
    ) where
        T: coeus_core::Float,
    {
        attention::sdp_attention(attention::AttentionForward {
            backend: self,
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            key_padding_mask,
            key_padding_mask_layout,
            is_causal,
            scale,
            output,
            output_layout,
            attn_weights,
            attn_weights_layout,
        });
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn sdp_attention_backward(
        &self,
        grad_out: &Self::DeviceBuffer<T>,
        _grad_out_layout: &Layout,
        query: &Self::DeviceBuffer<T>,
        query_layout: &Layout,
        key: &Self::DeviceBuffer<T>,
        key_layout: &Layout,
        value: &Self::DeviceBuffer<T>,
        value_layout: &Layout,
        attn_weights: &Self::DeviceBuffer<T>,
        _attn_weights_layout: &Layout,
        scale: T,
        grad_q: Option<&mut Self::DeviceBuffer<T>>,
        grad_k: Option<&mut Self::DeviceBuffer<T>>,
        grad_v: Option<&mut Self::DeviceBuffer<T>>,
    ) where
        T: coeus_core::Float,
    {
        attention::sdp_attention_backward(attention::AttentionBackward {
            grad_out,
            query,
            query_layout,
            key,
            key_layout,
            value,
            value_layout,
            attn_weights,
            scale,
            grad_q,
            grad_k,
            grad_v,
        });
    }

    #[inline]
    fn sgd_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) where
        T: coeus_core::Float,
    {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_sgd_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &velocity.buffer,
            velocity_layout,
            lr,
            momentum,
            len,
        );
    }

    #[inline]
    fn adam_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adam_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &m.buffer,
            m_layout,
            &v.buffer,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            t,
            len,
        );
    }

    #[inline]
    fn rmsprop_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_rmsprop_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &v.buffer,
            v_layout,
            lr,
            alpha,
            eps,
            len,
        );
    }

    #[inline]
    fn adamw_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        t: usize,
    ) where
        T: coeus_core::Float,
    {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adamw_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &m.buffer,
            m_layout,
            &v.buffer,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
            len,
        );
    }

    #[inline]
    fn adagrad_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        history: &mut Self::DeviceBuffer<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) where
        T: coeus_core::Float,
    {
        let len = param_layout.shape().iter().product::<usize>();
        kernels::dispatch_adagrad_step::<T>(
            &param.buffer,
            param_layout,
            &grad.buffer,
            grad_layout,
            &history.buffer,
            history_layout,
            lr,
            eps,
            len,
        );
    }
}
