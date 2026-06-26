use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::Result;
use leto_ops::Scalar as LetoScalar;

/// 2D matrix multiplication of two coeus CPU tensors into caller-owned output.
/// Strided/transposed inputs are handled by the leto kernel without copies.
///
/// # Examples
///
/// `[[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]`:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::matmul_into;
///
/// let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let b = [7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0];
/// let mut out = [0.0_f64; 4];
/// matmul_into(
///     &Layout::new([2, 3].into()),
///     &a,
///     &Layout::new([3, 2].into()),
///     &b,
///     &Layout::new([2, 2].into()),
///     &mut out,
/// )
/// .unwrap();
/// assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
/// ```
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

/// Rank-3 batched matrix multiplication of coeus CPU tensors into caller-owned
/// output. The batch dimension of either input may be one and is broadcast by
/// Leto at zero stride.
///
/// # Examples
///
/// Two `[2,2,3]` batches multiply a broadcast `[1,3,2]` right-hand side,
/// producing a `[2,2,2]` batched result:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::batched_matmul_into;
///
/// let lhs = [
///     1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
///     7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
/// ];
/// let rhs = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0]; // single batch, broadcast
/// let mut out = [0.0_f64; 8];
/// batched_matmul_into(
///     &Layout::new([2, 2, 3].into()),
///     &lhs,
///     &Layout::new([1, 3, 2].into()),
///     &rhs,
///     &Layout::new([2, 2, 2].into()),
///     &mut out,
/// )
/// .unwrap();
/// assert_eq!(out, [45.0, 56.0, 99.0, 125.0, 153.0, 194.0, 207.0, 263.0]);
/// ```
pub fn batched_matmul_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let a_view = to_leto_view::<T, 3>(a_layout, a)?;
    let b_view = to_leto_view::<T, 3>(b_layout, b)?;
    let mut out_view = to_leto_view_mut::<T, 3>(out_layout, out)?;
    leto_ops::batched_matmul(&a_view, &b_view, &mut out_view)
}

/// 2D matrix multiplication with accumulation: `out += a * b`.
///
/// # Examples
///
/// The product `[[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]` is added
/// onto a pre-seeded output rather than overwriting it:
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::matmul_accumulate_into;
///
/// let a = [1.0_f64, 2.0, 3.0, 4.0];
/// let b = [5.0_f64, 6.0, 7.0, 8.0];
/// let mut out = [1.0_f64; 4]; // pre-seeded
/// let l = Layout::new([2, 2].into());
/// matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
/// assert_eq!(out, [20.0, 23.0, 44.0, 51.0]);
/// ```
pub fn matmul_accumulate_into<T: LetoScalar>(
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
    leto_ops::matmul_accumulate(&a_view, &b_view, &mut out_view)
}

/// Rank-3 batched matrix multiplication with accumulation: `out += a * b`.
///
/// # Examples
///
/// Each batch accumulates independently onto a pre-seeded output. Batch 0 is
/// the identity (so `out` gains `B0`); batch 1 is `2*I` (so `out` gains `2*B1`):
///
/// ```
/// use coeus_core::Layout;
/// use coeus_leto::batched_matmul_accumulate_into;
///
/// let a = [
///     1.0_f64, 0.0, 0.0, 1.0, // batch 0: identity
///     2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
/// ];
/// let b = [
///     5.0_f64, 6.0, 7.0, 8.0, // batch 0
///     1.0, 1.0, 1.0, 1.0, // batch 1
/// ];
/// let mut out = [1.0_f64; 8]; // pre-seeded
/// let l = Layout::new([2, 2, 2].into());
/// batched_matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
/// assert_eq!(out, [6.0, 7.0, 8.0, 9.0, 3.0, 3.0, 3.0, 3.0]);
/// ```
pub fn batched_matmul_accumulate_into<T: LetoScalar>(
    a_layout: &CoeusLayout,
    a: &[T],
    b_layout: &CoeusLayout,
    b: &[T],
    out_layout: &CoeusLayout,
    out: &mut [T],
) -> Result<()> {
    let shape_a: [usize; 3] =
        a_layout
            .shape()
            .try_into()
            .map_err(|_| leto::LetoError::ShapeMismatch {
                lhs: a_layout.shape().to_vec(),
                rhs: vec![],
            })?;
    let [lhs_batch, m, lhs_k] = shape_a;
    let shape_b: [usize; 3] =
        b_layout
            .shape()
            .try_into()
            .map_err(|_| leto::LetoError::ShapeMismatch {
                lhs: b_layout.shape().to_vec(),
                rhs: vec![],
            })?;
    let [rhs_batch, rhs_k, n] = shape_b;
    let shape_out: [usize; 3] =
        out_layout
            .shape()
            .try_into()
            .map_err(|_| leto::LetoError::ShapeMismatch {
                lhs: out_layout.shape().to_vec(),
                rhs: vec![],
            })?;
    let [out_batch, out_m, out_n] = shape_out;

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(leto::LetoError::ShapeMismatch {
            lhs: a_layout.shape().to_vec(),
            rhs: b_layout.shape().to_vec(),
        });
    }

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        a_layout.strides()[0] as isize
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        b_layout.strides()[0] as isize
    };
    let out_batch_stride = out_layout.strides()[0] as isize;

    let lhs_offset = a_layout.offset() as isize;
    let rhs_offset = b_layout.offset() as isize;
    let out_offset = out_layout.offset() as isize;

    for b_idx in 0..batch {
        let a_sub = CoeusLayout::from_shape_strides(
            coeus_core::Shape::from([m, lhs_k].as_slice()),
            coeus_core::Strides::from([a_layout.strides()[1], a_layout.strides()[2]].as_slice()),
            (lhs_offset + b_idx as isize * lhs_batch_stride) as usize,
        );
        let b_sub = CoeusLayout::from_shape_strides(
            coeus_core::Shape::from([rhs_k, n].as_slice()),
            coeus_core::Strides::from([b_layout.strides()[1], b_layout.strides()[2]].as_slice()),
            (rhs_offset + b_idx as isize * rhs_batch_stride) as usize,
        );
        let out_sub = CoeusLayout::from_shape_strides(
            coeus_core::Shape::from([out_m, out_n].as_slice()),
            coeus_core::Strides::from(
                [out_layout.strides()[1], out_layout.strides()[2]].as_slice(),
            ),
            (out_offset + b_idx as isize * out_batch_stride) as usize,
        );
        matmul_accumulate_into(&a_sub, a, &b_sub, b, &out_sub, out)?;
    }
    Ok(())
}
