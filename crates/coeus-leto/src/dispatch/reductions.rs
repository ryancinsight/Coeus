use crate::convert::{to_leto_layout, to_leto_view, to_leto_view_mut};
use coeus_core::{FloatElement, Layout as CoeusLayout, ReductionOp};
use leto::{
    application::{argmax, argmin},
    Array, LetoError, RankMarker, RemoveAxis, Result, SliceStorage, Storage,
};
use leto_ops::{
    CumProdOp, CumSumOp, MaxAxis, MeanAxis, MinAxis, ProductAxis, Scalar as LetoScalar,
    ScanDirection, ScanOp, SumAxis,
};

use super::MAX_DISPATCH_RANK;

#[inline(always)]
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
        ReductionOp::Prod => {
            leto_ops::reduce_axis_into::<ProductAxis, T, N>(&a_view, axis, &mut out_view)
        }
        ReductionOp::Mean => Err(LetoError::InvalidInput(
            "mean is defined only for floating-point elements; use mean_into".to_string(),
        )),
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
///
/// Mean is float-only and dispatches through [`mean_into`], whose
/// `FloatElement` bound rejects integer element types at compile time.
///
/// # Errors
///
/// Returns [`LetoError::InvalidInput`] for [`ReductionOp::Mean`], and the leto
/// layout or storage error when the rank is outside `1..=6` or the layouts do
/// not describe a keep-dim reduction along `axis`.
///
/// # Examples
///
/// Reduce a `[2,3]` matrix along axis 1 into a `[2,1]` keep-dim output, for the
/// `sum`, `product`, `max`, and `min` operators:
///
/// ```
/// use coeus_core::{Layout, ReductionOp};
/// use coeus_leto::reduce_into;
///
/// let input = [1.0_f64, 4.0, -2.0, 5.0, 3.0, 6.0];
/// let input_layout = Layout::new([2, 3].into());
/// let output_layout = Layout::new([2, 1].into());
/// let mut out = [0.0_f64; 2];
///
/// reduce_into(ReductionOp::Sum, &input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [3.0, 14.0]);
///
/// reduce_into(ReductionOp::Prod, &input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [-8.0, 90.0]);
///
/// let mean = reduce_into(ReductionOp::Mean, &input_layout, &input, 1, &output_layout, &mut out);
/// assert!(matches!(mean, Err(leto::LetoError::InvalidInput(_))));
///
/// reduce_into(ReductionOp::Max, &input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [4.0, 6.0]);
///
/// reduce_into(ReductionOp::Min, &input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [-2.0, 3.0]);
/// ```
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
        6 => reduce_n::<T, 6>(op, a_layout, a, axis, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn mean_n<T: LetoScalar + FloatElement, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    leto_ops::reduce_axis_into::<MeanAxis, T, N>(&a_view, axis, &mut out_view)
}

/// Keep-dim arithmetic mean of a coeus CPU tensor along `axis` into
/// caller-owned output, dispatched to the matching monomorphized leto
/// reduction kernel.
///
/// The `FloatElement` bound makes integer mean unrepresentable: integer
/// division would truncate the quotient, so integer callers convert to a
/// floating-point tensor first.
///
/// # Errors
///
/// Returns the leto layout or storage error when the rank is outside `1..=6`,
/// the layouts do not describe a keep-dim reduction along `axis`, or the axis
/// is empty while the output is not.
///
/// # Examples
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::mean_into;
///
/// let input = [1.0_f64, 4.0, -2.0, 5.0, 3.0, 6.0];
/// let input_layout = Layout::new([2, 3].into());
/// let output_layout = Layout::new([2, 1].into());
/// let mut out = [0.0_f64; 2];
/// mean_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [1.0, 14.0 / 3.0]);
/// ```
pub fn mean_into<T: LetoScalar + FloatElement>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    match a_layout.ndim() {
        1 => mean_n::<T, 1>(a_layout, a, axis, out_layout, out),
        2 => mean_n::<T, 2>(a_layout, a, axis, out_layout, out),
        3 => mean_n::<T, 3>(a_layout, a, axis, out_layout, out),
        4 => mean_n::<T, 4>(a_layout, a, axis, out_layout, out),
        5 => mean_n::<T, 5>(a_layout, a, axis, out_layout, out),
        6 => mean_n::<T, 6>(a_layout, a, axis, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

fn scan_n<Op, T, const N: usize>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    direction: ScanDirection,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()>
where
    Op: ScanOp<T>,
    T: LetoScalar,
{
    let a_view = to_leto_view::<T, N>(a_layout, a)?;
    let mut out_view = to_leto_view_mut::<T, N>(out_layout, out)?;
    leto_ops::scan_axis_into::<Op, T, N>(&a_view, axis, direction, &mut out_view)
}

fn scan_into<Op, T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    direction: ScanDirection,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()>
where
    Op: ScanOp<T>,
{
    match a_layout.ndim() {
        1 => scan_n::<Op, T, 1>(a_layout, a, axis, direction, out_layout, out),
        2 => scan_n::<Op, T, 2>(a_layout, a, axis, direction, out_layout, out),
        3 => scan_n::<Op, T, 3>(a_layout, a, axis, direction, out_layout, out),
        4 => scan_n::<Op, T, 4>(a_layout, a, axis, direction, out_layout, out),
        5 => scan_n::<Op, T, 5>(a_layout, a, axis, direction, out_layout, out),
        6 => scan_n::<Op, T, 6>(a_layout, a, axis, direction, out_layout, out),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Forward inclusive cumulative sum of a coeus CPU tensor into caller-owned
/// output, dispatched to the matching monomorphized leto scan kernel.
///
/// # Examples
///
/// Forward cumulative sum along axis 1 of a `[2,3]` matrix:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::cumsum_into;
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let layout = Layout::new([2, 3].into());
/// let mut out = [0.0_f64; 6];
/// cumsum_into(&layout, &input, 1, &layout, &mut out).unwrap();
/// assert_eq!(out, [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
/// ```
pub fn cumsum_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    scan_into::<CumSumOp, T>(a_layout, a, axis, ScanDirection::Forward, out_layout, out)
}

/// Reverse inclusive cumulative sum of a coeus CPU tensor into caller-owned
/// output, dispatched to the matching monomorphized leto scan kernel.
///
/// # Examples
///
/// Reverse cumulative sum (suffix sum) along axis 1 of a `[2,3]` matrix:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::suffix_sum_into;
///
/// let input = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let layout = Layout::new([2, 3].into());
/// let mut out = [0.0_f64; 6];
/// suffix_sum_into(&layout, &input, 1, &layout, &mut out).unwrap();
/// assert_eq!(out, [6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);
/// ```
pub fn suffix_sum_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    scan_into::<CumSumOp, T>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out)
}

/// Forward inclusive cumulative product of a coeus CPU tensor into
/// caller-owned output, dispatched to the matching monomorphized leto scan
/// kernel.
pub fn cumprod_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    scan_into::<CumProdOp, T>(a_layout, a, axis, ScanDirection::Forward, out_layout, out)
}

/// Reverse inclusive cumulative product of a coeus CPU tensor into
/// caller-owned output, dispatched to the matching monomorphized leto scan
/// kernel.
pub fn suffix_prod_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    axis: usize,
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    scan_into::<CumProdOp, T>(a_layout, a, axis, ScanDirection::Reverse, out_layout, out)
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
        argmax::<T, _, N, M>(&input, axis)?
    } else {
        argmin::<T, _, N, M>(&input, axis)?
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
///
/// # Examples
///
/// Argmax along axis 1 of a `[2,3]` matrix into a `[2,1]` keep-dim output:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::argmax_into;
///
/// let input = [1.0_f64, 4.0, -2.0, 5.0, 3.0, 6.0];
/// let input_layout = Layout::new([2, 3].into());
/// let output_layout = Layout::new([2, 1].into());
/// let mut out = [0_i64; 2];
/// argmax_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [1, 2]);
/// ```
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
        6 => arg_reduce_n::<T, 6, 5>(a_layout, a, axis, out_layout, out, true),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}

/// Keep-dim argmin of a coeus CPU tensor into caller-owned output, dispatched
/// to the matching monomorphized leto arg-reduction kernel.
///
/// # Examples
///
/// Argmin along axis 1 of a `[2,3]` matrix into a `[2,1]` keep-dim output:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::argmin_into;
///
/// let input = [1.0_f64, 4.0, -2.0, 5.0, 3.0, 6.0];
/// let input_layout = Layout::new([2, 3].into());
/// let output_layout = Layout::new([2, 1].into());
/// let mut out = [0_i64; 2];
/// argmin_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
/// assert_eq!(out, [2, 1]);
/// ```
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
        6 => arg_reduce_n::<T, 6, 5>(a_layout, a, axis, out_layout, out, false),
        n => Err(LetoError::StorageError {
            reason: format!("coeus-leto dispatch supports rank 1..={MAX_DISPATCH_RANK}, got {n}"),
        }),
    }
}
