use super::support::layout;
use super::{argmax_into, argmin_into, cumsum_into, reduce_into, suffix_sum_into, ReductionOp};

#[test]
fn reduction_dispatch_covers_keepdim_axis_ops() {
    let input = vec![1.0f64, 4.0, -2.0, 5.0, 3.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let output_layout = layout(&[2, 1]);
    let mut out = vec![0.0f64; 2];

    reduce_into(
        ReductionOp::Sum,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![3.0, 14.0]);

    reduce_into(
        ReductionOp::Mean,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![1.0, 14.0 / 3.0]);

    reduce_into(
        ReductionOp::Max,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![4.0, 6.0]);

    reduce_into(
        ReductionOp::Min,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![-2.0, 3.0]);
}

#[test]
fn arg_reduction_dispatch_covers_keepdim_axis_ops() {
    let input = vec![1.0f64, 4.0, -2.0, 5.0, 3.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let output_layout = layout(&[2, 1]);
    let mut out = vec![0i64; 2];

    argmax_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![1, 2]);

    argmin_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![2, 1]);
}

#[test]
fn scan_dispatch_covers_forward_and_reverse_axis_ops() {
    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let mut out = vec![0.0f64; 6];

    cumsum_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

    suffix_sum_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);
}
