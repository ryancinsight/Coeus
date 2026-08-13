//! Value-semantic coverage for broadcast gradient reduction.
//!
//! `reduce_broadcast` sums a gradient back onto the shape of a broadcast input.
//! Each case below states the closed-form sum the reduction must produce, so a
//! defect in axis selection or axis ordering fails on values rather than shape
//! alone. The gradient is filled with distinct values (never all-ones) so an
//! omitted or duplicated reduction axis cannot coincide with the expected sum.

use coeus_autograd::backward::reduce_broadcast;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// Row-major gradient whose element at flat index `i` is `i + 1`.
fn ramp(shape: Vec<usize>) -> Tensor<f32, MoiraiBackend> {
    let backend = MoiraiBackend::new();
    let len = shape.iter().product::<usize>();
    let values = (0..len).map(|i| i as f32 + 1.0).collect::<Vec<_>>();
    Tensor::from_slice_on(shape, &values, &backend)
}

fn assert_reduces_to(grad_shape: Vec<usize>, target: &[usize], expected: &[f32]) {
    let reduced = reduce_broadcast(ramp(grad_shape.clone()), target);
    assert_eq!(
        reduced.shape(),
        target,
        "shape for {grad_shape:?} -> {target:?}"
    );
    assert_eq!(
        reduced.as_slice(),
        expected,
        "values for {grad_shape:?} -> {target:?}"
    );
}

#[test]
fn matching_shape_passes_the_gradient_through_unchanged() {
    assert_reduces_to(vec![2, 3], &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn leading_extra_dim_sums_over_the_batch_axis() {
    // [2,2,2] -> [2,2]: out[i] = grad[i] + grad[i + 4].
    assert_reduces_to(vec![2, 2, 2], &[2, 2], &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn aligned_unit_axis_sums_across_columns() {
    // [2,3] -> [2,1]: row sums 1+2+3 and 4+5+6.
    assert_reduces_to(vec![2, 3], &[2, 1], &[6.0, 15.0]);
}

#[test]
fn aligned_unit_axis_sums_across_rows() {
    // [2,3] -> [1,3]: column sums (1+4, 2+5, 3+6).
    assert_reduces_to(vec![2, 3], &[1, 3], &[5.0, 7.0, 9.0]);
}

#[test]
fn trailing_unit_axis_reduces_after_an_earlier_axis_was_already_reduced() {
    // Both aligned axes are unit in the target, so the reduction at axis 1 must
    // still fire after axis 0 has been collapsed to extent 1 — the ordering the
    // running-shape predicate has to get right.
    assert_reduces_to(vec![2, 3], &[1, 1], &[21.0]);
}

#[test]
fn leading_extra_dim_and_aligned_unit_axis_compose() {
    // [2,2,3] -> [2,1]: sum the leading axis, then the trailing axis.
    // Leading sum: [1+7, 2+8, 3+9; 4+10, 5+11, 6+12] = [8,10,12; 14,16,18].
    // Row sums: 30 and 48.
    assert_reduces_to(vec![2, 2, 3], &[2, 1], &[30.0, 48.0]);
}

#[test]
fn full_reduction_to_a_scalar_shape_sums_every_element() {
    assert_reduces_to(vec![2, 3], &[1], &[21.0]);
}

#[test]
fn non_broadcast_axes_are_left_untouched() {
    // Target extents match the gradient on every aligned axis, so only the
    // leading extra axis reduces; no aligned axis may be summed away.
    assert_reduces_to(vec![2, 1, 3], &[1, 3], &[5.0, 7.0, 9.0]);
}
