use crate::convert::{to_leto_layout, to_leto_view, to_leto_view_mut};
use coeus_core::{
    BinaryOp, CpuUnaryOp as UnaryOp, Layout as CoeusLayout, ReductionOp, Scalar as CoeusScalar,
};
use leto::{Array, LetoError, PadWidth, RankMarker, RemoveAxis, Result, SliceStorage, Storage};
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

fn arg_reduce_n<T: LetoScalar, const N: usize, const M: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [i64],
    largest: bool,
) -> Result<()>
where
    RankMarker<N>: RemoveAxis<N, SmallerShape = [usize; M], SmallerStrides = [isize; M]>,
{
    let mut expected_shape = a_layout.shape().to_vec();
    if axis >= expected_shape.len() {
        return Err(LetoError::StorageError {
            reason: format!("axis {axis} out of bounds for rank {N}"),
        });
    }
    expected_shape[axis] = 1;
    if out_layout.shape() != expected_shape.as_slice() {
        return Err(LetoError::ShapeMismatch {
            lhs: expected_shape,
            rhs: out_layout.shape().to_vec(),
        });
    }

    let input = Array::<T, _, N>::new(to_leto_layout::<N>(a_layout)?, SliceStorage::new(a))?;
    let reduced = if largest {
        leto::application::argmax::<T, _, N, M>(&input, axis)?
    } else {
        leto::application::argmin::<T, _, N, M>(&input, axis)?
    };
    write_keepdim_indices::<N, M>(
        a_layout.shape(),
        axis,
        reduced.storage().as_slice(),
        out_layout,
        out,
    )
}

fn write_keepdim_indices<const N: usize, const M: usize>(
    input_shape: &[usize],
    axis: usize,
    reduced: &[usize],
    out_layout: &CoeusLayout,
    out: &mut [i64],
) -> Result<()> {
    let reduced_size: usize = input_shape
        .iter()
        .enumerate()
        .filter_map(|(dim, &len)| (dim != axis).then_some(len))
        .product();
    if reduced.len() != reduced_size {
        return Err(LetoError::StorageError {
            reason: format!(
                "arg reduction produced {} values, expected {reduced_size}",
                reduced.len()
            ),
        });
    }

    for (flat, &index) in reduced.iter().enumerate() {
        let mut rem = flat;
        let mut coords = [0usize; MAX_DISPATCH_RANK];
        for dim in (0..N).rev() {
            if dim == axis {
                coords[dim] = 0;
                continue;
            }
            let len = input_shape[dim];
            coords[dim] = rem % len;
            rem /= len;
        }
        let out_offset = out_layout.physical_index(&coords[..N]);
        out[out_offset] = i64::try_from(index).map_err(|_| LetoError::StorageError {
            reason: format!("axis index {index} exceeds i64 range"),
        })?;
    }

    Ok(())
}

/// Keep-dim argmax of a coeus CPU tensor into caller-owned output, dispatched
/// to the matching monomorphized leto arg-reduction kernel.
pub fn argmax_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [i64],
) -> Result<()> {
    match a_layout.ndim() {
        1 => arg_reduce_n::<T, 1, 0>(a_layout, a, axis, out_layout, out, true),
        2 => arg_reduce_n::<T, 2, 1>(a_layout, a, axis, out_layout, out, true),
        3 => arg_reduce_n::<T, 3, 2>(a_layout, a, axis, out_layout, out, true),
        4 => arg_reduce_n::<T, 4, 3>(a_layout, a, axis, out_layout, out, true),
        5 => arg_reduce_n::<T, 5, 4>(a_layout, a, axis, out_layout, out, true),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Keep-dim argmin of a coeus CPU tensor into caller-owned output, dispatched
/// to the matching monomorphized leto arg-reduction kernel.
pub fn argmin_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [i64],
) -> Result<()> {
    match a_layout.ndim() {
        1 => arg_reduce_n::<T, 1, 0>(a_layout, a, axis, out_layout, out, false),
        2 => arg_reduce_n::<T, 2, 1>(a_layout, a, axis, out_layout, out, false),
        3 => arg_reduce_n::<T, 3, 2>(a_layout, a, axis, out_layout, out, false),
        4 => arg_reduce_n::<T, 4, 3>(a_layout, a, axis, out_layout, out, false),
        5 => arg_reduce_n::<T, 5, 4>(a_layout, a, axis, out_layout, out, false),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn pad_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    widths: &[(usize, usize)],
    fill: T,
) -> Result<Vec<T>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    let width: PadWidth<N> =
        widths.try_into().map_err(
            |_: std::array::TryFromSliceError| LetoError::ShapeMismatch {
                lhs: vec![N],
                rhs: vec![widths.len()],
            },
        )?;
    let padded = leto::application::pad(&input, width, fill)?;
    Ok(padded.storage().as_slice().to_vec())
}

/// Constant padding of a coeus CPU tensor, dispatched from dynamic rank to the
/// matching monomorphized leto structural kernel. The returned values are
/// C-contiguous in row-major output order.
pub fn pad_values<T: Clone>(
    a_layout: &CoeusLayout,
    a: &[T],
    widths: &[(usize, usize)],
    fill: T,
) -> Result<Vec<T>> {
    match a_layout.ndim() {
        1 => pad_values_n::<T, 1>(a_layout, a, widths, fill),
        2 => pad_values_n::<T, 2>(a_layout, a, widths, fill),
        3 => pad_values_n::<T, 3>(a_layout, a, widths, fill),
        4 => pad_values_n::<T, 4>(a_layout, a, widths, fill),
        5 => pad_values_n::<T, 5>(a_layout, a, widths, fill),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
