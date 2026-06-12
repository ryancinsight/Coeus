use crate::convert::{to_leto_layout, to_leto_view, to_leto_view_mut};
use coeus_core::{
    BinaryOp, CpuUnaryOp as UnaryOp, Layout as CoeusLayout, ReductionOp, Scalar as CoeusScalar,
};
use leto::{Array, LetoError, PadWidth, RankMarker, RemoveAxis, Result, SliceStorage, Storage};
use leto_ops::{
    CumSumOp, MaxAxis, MeanAxis, MinAxis, RealScalar, Scalar as LetoScalar, ScanDirection, SumAxis,
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

fn concat_values_n<T: Clone, const N: usize>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>> {
    if layouts.len() != inputs.len() {
        return Err(LetoError::StorageError {
            reason: format!(
                "concat received {} layouts for {} inputs",
                layouts.len(),
                inputs.len()
            ),
        });
    }

    let mut views = Vec::with_capacity(inputs.len());
    for (&layout, &input) in layouts.iter().zip(inputs) {
        views.push(to_leto_view::<T, N>(layout, input)?);
    }

    let concatenated = leto::application::concat(&views, axis)?;
    Ok(concatenated.storage().as_slice().to_vec())
}

/// Concatenate coeus CPU tensor values along `axis`, dispatched from dynamic
/// rank to the matching monomorphized leto structural kernel. The returned
/// values are C-contiguous in row-major output order.
pub fn concat_values<T: Clone>(
    layouts: &[&CoeusLayout],
    inputs: &[&[T]],
    axis: usize,
) -> Result<Vec<T>> {
    let Some(first) = layouts.first() else {
        return Err(LetoError::StorageError {
            reason: "concat requires at least one input".to_string(),
        });
    };
    match first.ndim() {
        1 => concat_values_n::<T, 1>(layouts, inputs, axis),
        2 => concat_values_n::<T, 2>(layouts, inputs, axis),
        3 => concat_values_n::<T, 3>(layouts, inputs, axis),
        4 => concat_values_n::<T, 4>(layouts, inputs, axis),
        5 => concat_values_n::<T, 5>(layouts, inputs, axis),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn split_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<T>>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    let views = leto::application::split(&input, axis, sizes)?;
    Ok(views
        .iter()
        .map(|view| view.to_contiguous().storage().as_slice().to_vec())
        .collect())
}

/// Split coeus CPU tensor values along `axis`, dispatched from dynamic rank to
/// the matching monomorphized leto structural kernel. Each returned chunk is
/// C-contiguous in row-major output order.
pub fn split_values<T: Clone>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<T>>> {
    match a_layout.ndim() {
        1 => split_values_n::<T, 1>(a_layout, a, axis, sizes),
        2 => split_values_n::<T, 2>(a_layout, a, axis, sizes),
        3 => split_values_n::<T, 3>(a_layout, a, axis, sizes),
        4 => split_values_n::<T, 4>(a_layout, a, axis, sizes),
        5 => split_values_n::<T, 5>(a_layout, a, axis, sizes),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn contiguous_values_n<T: Clone, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
) -> Result<Vec<T>> {
    let input = to_leto_view::<T, N>(a_layout, a)?;
    Ok(input.to_contiguous().storage().as_slice().to_vec())
}

/// Materialize a coeus CPU tensor view into C-contiguous row-major values,
/// dispatched from dynamic rank to Leto's const-rank view materializer.
pub fn contiguous_values<T: Clone>(a_layout: &CoeusLayout, a: &[T]) -> Result<Vec<T>> {
    match a_layout.ndim() {
        1 => contiguous_values_n::<T, 1>(a_layout, a),
        2 => contiguous_values_n::<T, 2>(a_layout, a),
        3 => contiguous_values_n::<T, 3>(a_layout, a),
        4 => contiguous_values_n::<T, 4>(a_layout, a),
        5 => contiguous_values_n::<T, 5>(a_layout, a),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn from_leto_layout<const N: usize>(layout: leto::Layout<N>) -> Result<CoeusLayout> {
    let strides = layout
        .strides
        .iter()
        .map(|&stride| {
            usize::try_from(stride).map_err(|_| LetoError::StorageError {
                reason: format!("coeus layout cannot represent negative stride {stride}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CoeusLayout::from_shape_strides(
        layout.shape.to_vec().into(),
        strides.as_slice().into(),
        layout.offset,
    ))
}

fn reshape_layout_n_m<const N: usize, const M: usize>(
    layout: &CoeusLayout,
    shape: &[usize],
) -> Result<CoeusLayout> {
    let reshaped = to_leto_layout::<N>(layout)?.reshape::<M>(shape_n::<M>(shape)?)?;
    from_leto_layout(reshaped)
}

fn reshape_layout_n<const N: usize>(layout: &CoeusLayout, shape: &[usize]) -> Result<CoeusLayout> {
    match shape.len() {
        1 => reshape_layout_n_m::<N, 1>(layout, shape),
        2 => reshape_layout_n_m::<N, 2>(layout, shape),
        3 => reshape_layout_n_m::<N, 3>(layout, shape),
        4 => reshape_layout_n_m::<N, 4>(layout, shape),
        5 => reshape_layout_n_m::<N, 5>(layout, shape),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Zero-copy reshape of coeus layout metadata, dispatched to Leto's const-rank
/// layout validator. This preserves the storage offset and returns a dynamic
/// coeus layout for the requested shape.
pub fn reshape_layout(layout: &CoeusLayout, shape: &[usize]) -> Result<CoeusLayout> {
    match layout.ndim() {
        1 => reshape_layout_n::<1>(layout, shape),
        2 => reshape_layout_n::<2>(layout, shape),
        3 => reshape_layout_n::<3>(layout, shape),
        4 => reshape_layout_n::<4>(layout, shape),
        5 => reshape_layout_n::<5>(layout, shape),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn permute_layout_n<const N: usize>(layout: &CoeusLayout, axes: &[usize]) -> Result<CoeusLayout> {
    let permuted = to_leto_layout::<N>(layout)?.transpose(shape_n::<N>(axes)?)?;
    from_leto_layout(permuted)
}

/// Zero-copy permutation of coeus layout metadata, dispatched to Leto's
/// const-rank layout validator.
pub fn permute_layout(layout: &CoeusLayout, axes: &[usize]) -> Result<CoeusLayout> {
    match layout.ndim() {
        1 => permute_layout_n::<1>(layout, axes),
        2 => permute_layout_n::<2>(layout, axes),
        3 => permute_layout_n::<3>(layout, axes),
        4 => permute_layout_n::<4>(layout, axes),
        5 => permute_layout_n::<5>(layout, axes),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn from_shape_fn_values_n<T: Clone, F, const N: usize>(shape: &[usize], f: &F) -> Result<Vec<T>>
where
    F: Fn(&[usize]) -> T,
{
    let values = Array::<T, _, N>::from_shape_fn(shape_n::<N>(shape)?, |index| f(&index));
    Ok(values.storage().as_slice().to_vec())
}

/// Generate C-contiguous row-major values for a coeus dynamic-rank shape,
/// dispatched to Leto's const-rank coordinate generator.
pub fn from_shape_fn_values<T: Clone, F>(shape: &[usize], f: F) -> Result<Vec<T>>
where
    F: Fn(&[usize]) -> T,
{
    match shape.len() {
        1 => from_shape_fn_values_n::<T, F, 1>(shape, &f),
        2 => from_shape_fn_values_n::<T, F, 2>(shape, &f),
        3 => from_shape_fn_values_n::<T, F, 3>(shape, &f),
        4 => from_shape_fn_values_n::<T, F, 4>(shape, &f),
        5 => from_shape_fn_values_n::<T, F, 5>(shape, &f),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn shape_n<const N: usize>(shape: &[usize]) -> Result<[usize; N]> {
    shape.try_into().map_err(
        |_: std::array::TryFromSliceError| LetoError::ShapeMismatch {
            lhs: vec![N],
            rhs: vec![shape.len()],
        },
    )
}

fn uniform_values_n<T: RealScalar, const N: usize>(
    shape: &[usize],
    low: T,
    high: T,
    seed: u64,
) -> Result<Vec<T>> {
    let values = leto_ops::uniform_with_seed(shape_n::<N>(shape)?, low, high, seed)?;
    Ok(values.storage().as_slice().to_vec())
}

/// Deterministic uniform initialization values for a coeus dynamic-rank shape,
/// dispatched to the matching monomorphized leto random constructor.
pub fn uniform_values<T: RealScalar>(
    shape: &[usize],
    low: T,
    high: T,
    seed: u64,
) -> Result<Vec<T>> {
    match shape.len() {
        1 => uniform_values_n::<T, 1>(shape, low, high, seed),
        2 => uniform_values_n::<T, 2>(shape, low, high, seed),
        3 => uniform_values_n::<T, 3>(shape, low, high, seed),
        4 => uniform_values_n::<T, 4>(shape, low, high, seed),
        5 => uniform_values_n::<T, 5>(shape, low, high, seed),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn normal_values_n<T: RealScalar, const N: usize>(
    shape: &[usize],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Vec<T>> {
    let values = leto_ops::normal_with_seed(shape_n::<N>(shape)?, mean, std_dev, seed)?;
    Ok(values.storage().as_slice().to_vec())
}

/// Deterministic normal initialization values for a coeus dynamic-rank shape,
/// dispatched to the matching monomorphized leto random constructor.
pub fn normal_values<T: RealScalar>(
    shape: &[usize],
    mean: T,
    std_dev: T,
    seed: u64,
) -> Result<Vec<T>> {
    match shape.len() {
        1 => normal_values_n::<T, 1>(shape, mean, std_dev, seed),
        2 => normal_values_n::<T, 2>(shape, mean, std_dev, seed),
        3 => normal_values_n::<T, 3>(shape, mean, std_dev, seed),
        4 => normal_values_n::<T, 4>(shape, mean, std_dev, seed),
        5 => normal_values_n::<T, 5>(shape, mean, std_dev, seed),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
