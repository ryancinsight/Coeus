// ── Non-generic helper functions for layout indexing (Inner-Function Pattern) ──
// These compute physical offsets from logical coordinates, avoiding monomorphization bloat.

#[inline]
pub(crate) fn compute_broadcast_offsets(
    i: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    a_shape: &[usize],
    a_strides: &[usize],
    a_off: usize,
    b_shape: &[usize],
    b_strides: &[usize],
    b_off: usize,
) -> (usize, usize) {
    let mut temp = i;
    let mut off_a = a_off;
    let mut off_b = b_off;
    let ndim = out_shape.len();

    for d in 0..ndim {
        let coord = temp / out_strides[d];
        temp %= out_strides[d];

        if d >= ndim - a_shape.len() {
            let ad = d + a_shape.len() - ndim;
            if a_shape[ad] > 1 {
                off_a += coord * a_strides[ad];
            }
        }
        if d >= ndim - b_shape.len() {
            let bd = d + b_shape.len() - ndim;
            if b_shape[bd] > 1 {
                off_b += coord * b_strides[bd];
            }
        }
    }
    (off_a, off_b)
}

#[inline]
pub(crate) fn compute_unary_offset(
    i: usize,
    out_strides: &[usize],
    in_shape: &[usize],
    in_strides: &[usize],
    in_offset: usize,
) -> usize {
    let mut temp = i;
    let mut physical_index = in_offset;
    let ndim = in_shape.len();
    for d in 0..ndim {
        let coord = temp / out_strides[d];
        temp %= out_strides[d];
        physical_index += coord * in_strides[d];
    }
    physical_index
}

#[inline]
pub(crate) fn compute_reduction_base_offset(
    i: usize,
    out_strides: &[usize],
    a_shape: &[usize],
    a_strides: &[usize],
    a_off: usize,
    axis: usize,
) -> usize {
    let mut temp = i;
    let mut base_off_a = a_off;
    let ndim = a_shape.len();
    for d in 0..ndim {
        let coord = temp / out_strides[d];
        temp %= out_strides[d];
        if d != axis {
            base_off_a += coord * a_strides[d];
        }
    }
    base_off_a
}
