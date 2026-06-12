use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::{
    BinaryOp, CpuUnaryOp as UnaryOp, Layout as CoeusLayout, ReductionOp, Scalar as CoeusScalar,
};
use leto::{LetoError, Result};
use leto_ops::{
    CumSumOp, MaxAxis, MeanAxis, MinAxis, Scalar as LetoScalar, ScanDirection, SumAxis,
};

/// Largest dynamic rank the const-rank dispatch resolves. Coeus activations and
/// Apollo transforms stay well within this bound; ranks beyond it are a logged
/// error rather than silent truncation.
pub const MAX_DISPATCH_RANK: usize = 5;

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
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// 2D matrix multiplication of two coeus CPU tensors into caller-owned output.
/// Strided/transposed inputs are handled by the leto kernel without copies.
pub fn matmul_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, 2>(a_layout, a)?;
    let b_view = to_leto_view::<T, 2>(b_layout, b)?;
    let mut out_view = to_leto_view_mut::<T, 2>(out_layout, out)?;
    leto_ops::matmul(&a_view, &b_view, &mut out_view)
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
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn reduce_n<T: LetoScalar, const N: usize>(
    op: ReductionOp,
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    match op {
        ReductionOp::Sum => {
            leto_ops::reduce_axis_into::<SumAxis, T, N>(&a_view, axis, &mut out_view)
        }
        ReductionOp::Mean => {
            leto_ops::reduce_axis_into::<MeanAxis, T, N>(&a_view, axis, &mut out_view)
        }
        ReductionOp::Max => {
            leto_ops::reduce_axis_into::<MaxAxis, T, N>(&a_view, axis, &mut out_view)
        }
        ReductionOp::Min => {
            leto_ops::reduce_axis_into::<MinAxis, T, N>(&a_view, axis, &mut out_view)
        }
    }
}

/// Keep-dim axis reductions of a coeus CPU tensor into caller-owned output,
/// dispatched to the matching monomorphized leto reduction kernel.
pub fn reduce_into<T: LetoScalar>(
    op: ReductionOp,
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match a_layout.ndim() {
        1 => reduce_n::<T, 1>(op, a_layout, a, axis, out_layout, out),
        2 => reduce_n::<T, 2>(op, a_layout, a, axis, out_layout, out),
        3 => reduce_n::<T, 3>(op, a_layout, a, axis, out_layout, out),
        4 => reduce_n::<T, 4>(op, a_layout, a, axis, out_layout, out),
        5 => reduce_n::<T, 5>(op, a_layout, a, axis, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn scan_sum_n<T: LetoScalar, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    direction: ScanDirection,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    leto_ops::scan_axis_into::<CumSumOp, T, N>(&a_view, axis, direction, &mut out_view)
}

/// Forward inclusive cumulative sum of a coeus CPU tensor into caller-owned
/// output, dispatched to the matching monomorphized leto scan kernel.
pub fn cumsum_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match a_layout.ndim() {
        1 => scan_sum_n::<T, 1>(a_layout, a, axis, ScanDirection::Forward, out_layout, out),
        2 => scan_sum_n::<T, 2>(a_layout, a, axis, ScanDirection::Forward, out_layout, out),
        3 => scan_sum_n::<T, 3>(a_layout, a, axis, ScanDirection::Forward, out_layout, out),
        4 => scan_sum_n::<T, 4>(a_layout, a, axis, ScanDirection::Forward, out_layout, out),
        5 => scan_sum_n::<T, 5>(a_layout, a, axis, ScanDirection::Forward, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Reverse inclusive cumulative sum of a coeus CPU tensor into caller-owned
/// output, dispatched to the matching monomorphized leto scan kernel.
pub fn suffix_sum_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match a_layout.ndim() {
        1 => scan_sum_n::<T, 1>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out),
        2 => scan_sum_n::<T, 2>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out),
        3 => scan_sum_n::<T, 3>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out),
        4 => scan_sum_n::<T, 4>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out),
        5 => scan_sum_n::<T, 5>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
