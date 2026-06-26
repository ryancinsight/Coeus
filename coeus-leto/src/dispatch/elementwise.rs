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
///
/// # Examples
///
/// Add two `[2,2]` matrices, and broadcast a `[2,1]` column against a `[1,2]`
/// row into a `[2,2]` output without materializing the broadcasted operands:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::elementwise_add_into;
///
/// let la = Layout::new([2, 2].into());
/// let a = [1.0_f64, 2.0, 3.0, 4.0];
/// let b = [10.0_f64, 20.0, 30.0, 40.0];
/// let mut out = [0.0_f64; 4];
/// elementwise_add_into(&la, &a, &la, &b, &la, &mut out).unwrap();
/// assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
///
/// let col = Layout::new([2, 1].into());
/// let row = Layout::new([1, 2].into());
/// let out2 = Layout::new([2, 2].into());
/// let mut z = [0.0_f64; 4];
/// elementwise_add_into(&col, &[1.0, 2.0], &row, &[10.0, 20.0], &out2, &mut z).unwrap();
/// assert_eq!(z, [11.0, 21.0, 12.0, 22.0]);
/// ```
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
///
/// # Examples
///
/// Apply subtraction, multiplication, and division to two `[2,2]` matrices:
///
/// ```
/// use coeus_core::{BinaryOp, Layout};
/// use coeus_leto::elementwise_binary_into;
///
/// let la = Layout::new([2, 2].into());
/// let a = [8.0_f64, 9.0, 10.0, 12.0];
/// let b = [2.0_f64, 3.0, 5.0, 6.0];
/// let mut out = [0.0_f64; 4];
///
/// elementwise_binary_into(BinaryOp::Sub, &la, &a, &la, &b, &la, &mut out).unwrap();
/// assert_eq!(out, [6.0, 6.0, 5.0, 6.0]);
///
/// elementwise_binary_into(BinaryOp::Mul, &la, &a, &la, &b, &la, &mut out).unwrap();
/// assert_eq!(out, [16.0, 27.0, 50.0, 72.0]);
///
/// elementwise_binary_into(BinaryOp::Div, &la, &a, &la, &b, &la, &mut out).unwrap();
/// assert_eq!(out, [4.0, 3.0, 2.0, 2.0]);
/// ```
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
///
/// # Examples
///
/// Apply `relu`, `abs`, and `sqrt` to a `[2,2]` matrix:
///
/// ```
/// use coeus_core::{CpuUnaryOp, Layout};
/// use coeus_leto::elementwise_unary_into;
///
/// let la = Layout::new([2, 2].into());
/// let input = [-4.0_f64, -1.0, 0.0, 9.0];
/// let mut out = [0.0_f64; 4];
///
/// elementwise_unary_into(CpuUnaryOp::Relu, &la, &input, &la, &mut out).unwrap();
/// assert_eq!(out, [0.0, 0.0, 0.0, 9.0]);
///
/// elementwise_unary_into(CpuUnaryOp::Abs, &la, &input, &la, &mut out).unwrap();
/// assert_eq!(out, [4.0, 1.0, 0.0, 9.0]);
///
/// let squares = [0.0_f64, 1.0, 4.0, 16.0];
/// elementwise_unary_into(CpuUnaryOp::Sqrt, &la, &squares, &la, &mut out).unwrap();
/// assert_eq!(out, [0.0, 1.0, 2.0, 4.0]);
/// ```
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
