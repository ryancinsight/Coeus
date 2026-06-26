use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::{BinaryOp, CpuUnaryOp as UnaryOp, Layout as CoeusLayout, Scalar as CoeusScalar};
use leto::{LetoError, Result};
use leto_ops::Scalar as LetoScalar;

use super::MAX_DISPATCH_RANK;

/// Rank-`N` elementwise add into caller-owned output. Inputs broadcast to the
/// output shape through the leto kernel, so `[N,1]` + `[1,C]` -> `[N,C]` works
/// without materializing broadcasted operands.
fn add_n<T: LetoScalar, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let b_view = to_leto_view::<T, N>(b_layout, b)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    leto_ops::add(&a_view, &b_view, &mut out_view)
}

/// Elementwise add of two coeus CPU tensors into caller-owned output, dispatched
/// from the runtime rank to the matching monomorphized leto kernel.
pub fn elementwise_add_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match out_layout.ndim() {
        1 => add_n::<T, 1>(a_layout, a, b_layout, b, out_layout, out),
        2 => add_n::<T, 2>(a_layout, a, b_layout, b, out_layout, out),
        3 => add_n::<T, 3>(a_layout, a, b_layout, b, out_layout, out),
        4 => add_n::<T, 4>(a_layout, a, b_layout, b, out_layout, out),
        5 => add_n::<T, 5>(a_layout, a, b_layout, b, out_layout, out),
        6 => add_n::<T, 6>(a_layout, a, b_layout, b, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn binary_n<T: LetoScalar, const N: usize>(
    op: BinaryOp,
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let b_view = to_leto_view::<T, N>(b_layout, b)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    match op {
        BinaryOp::Add => leto_ops::add(&a_view, &b_view, &mut out_view),
        BinaryOp::Sub => leto_ops::sub(&a_view, &b_view, &mut out_view),
        BinaryOp::Mul => leto_ops::mul(&a_view, &b_view, &mut out_view),
        BinaryOp::Div => leto_ops::div(&a_view, &b_view, &mut out_view),
    }
}

/// Elementwise binary operations of two coeus CPU tensors into caller-owned output,
/// dispatched to the matching monomorphized leto kernel.
pub fn elementwise_binary_into<T: LetoScalar>(
    op: BinaryOp,
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match out_layout.ndim() {
        1 => binary_n::<T, 1>(op, a_layout, a, b_layout, b, out_layout, out),
        2 => binary_n::<T, 2>(op, a_layout, a, b_layout, b, out_layout, out),
        3 => binary_n::<T, 3>(op, a_layout, a, b_layout, b, out_layout, out),
        4 => binary_n::<T, 4>(op, a_layout, a, b_layout, b, out_layout, out),
        5 => binary_n::<T, 5>(op, a_layout, a, b_layout, b, out_layout, out),
        6 => binary_n::<T, 6>(op, a_layout, a, b_layout, b, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn unary_n<T: LetoScalar + CoeusScalar, const N: usize>(
    op: UnaryOp,
    a_layout: &CoeusLayout,
    a: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    leto_ops::map_into(&a_view, &mut out_view, move |x| T::eval_unary(op, x))
}

/// Elementwise unary operations of a coeus CPU tensor into caller-owned output,
/// dispatched to the matching monomorphized leto mapping kernel.
pub fn elementwise_unary_into<T: LetoScalar + CoeusScalar>(
    op: UnaryOp,
    a_layout: &CoeusLayout,
    a: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match out_layout.ndim() {
        1 => unary_n::<T, 1>(op, a_layout, a, out_layout, out),
        2 => unary_n::<T, 2>(op, a_layout, a, out_layout, out),
        3 => unary_n::<T, 3>(op, a_layout, a, out_layout, out),
        4 => unary_n::<T, 4>(op, a_layout, a, out_layout, out),
        5 => unary_n::<T, 5>(op, a_layout, a, out_layout, out),
        6 => unary_n::<T, 6>(op, a_layout, a, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
