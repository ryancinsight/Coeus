use super::support::layout;
use super::{
    argmax_into, argmin_into, cumprod_into, cumsum_into, reduce_into, reduce_mean_into,
    reduce_prod_into, suffix_prod_into, suffix_sum_into, ReductionOp,
};

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

    reduce_mean_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
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
fn float_only_reductions_have_their_own_bounded_entry_points() {
    // `reduce_into` refuses `Mean` and `Prod` because they need a
    // `FloatElement`; the bounded entry points are how a caller reaches them.
    let input = vec![1.0f64, 4.0, -2.0, 5.0, 3.0, 6.0];
    let input_layout = layout(&[2, 3]);
    let output_layout = layout(&[2, 1]);
    let mut out = vec![0.0f64; 2];

    reduce_mean_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 14.0 / 3.0]);

    reduce_prod_into(&input_layout, &input, 1, &output_layout, &mut out).unwrap();
    assert_eq!(out, vec![-8.0, 90.0]);

    // And the refusal names the call to make instead.
    let error = reduce_into(
        ReductionOp::Mean,
        &input_layout,
        &input,
        1,
        &output_layout,
        &mut out,
    )
    .expect_err("reduce_into must refuse the float-only ops");
    assert!(
        format!("{error:?}").contains("reduce_mean_into"),
        "the refusal must name the entry point to use: {error:?}"
    );
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

    cumprod_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);

    suffix_prod_into(&input_layout, &input, 1, &input_layout, &mut out).unwrap();
    assert_eq!(out, vec![6.0, 6.0, 3.0, 120.0, 30.0, 6.0]);
}
