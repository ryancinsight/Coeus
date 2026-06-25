use crate::convert::{to_leto_view, to_leto_view_mut};
use coeus_core::Layout as CoeusLayout;
use leto::Result;
use leto_ops::Scalar as LetoScalar;

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

/// Rank-3 batched matrix multiplication of coeus CPU tensors into caller-owned
/// output. The batch dimension of either input may be one and is broadcast by
/// Leto at zero stride.
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
