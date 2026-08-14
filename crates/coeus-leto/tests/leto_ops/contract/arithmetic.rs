#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use super::support::layout;
use super::{
    elementwise_add_into, elementwise_binary_into, elementwise_unary_into, BinaryOp, CpuUnaryOp,
};

fn assert_transcendental_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        if expected.is_infinite() {
            assert_eq!(actual, expected);
            continue;
        }
        // The provider and scalar reference may select independent libm
        // implementations. Four epsilon-scaled units budget one final
        // rounding in each path plus one guard unit per implementation.
        let tolerance = 4.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }
}

#[test]
fn add_matches_reference_rank2() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![10.0f64, 20.0, 30.0, 40.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_add_into(&la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_broadcasts_rowvec_into_matrix() {
    // [2,1] + [1,2] -> [2,2], exercising the broadcast-aware leto kernel from
    // coeus's dynamic-rank entry point.
    let a = vec![1.0f64, 2.0]; // shape [2,1]
    let b = vec![10.0f64, 20.0]; // shape [1,2]
    let mut out = vec![0.0f64; 4];

    elementwise_add_into(
        &layout(&[2, 1]),
        &a,
        &layout(&[1, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // rows: [1+10, 1+20], [2+10, 2+20]
    assert_eq!(out, vec![11.0, 21.0, 12.0, 22.0]);
}

#[test]
fn comparisons_broadcast_row_into_matrix() {
    let lhs_layout = layout(&[2, 3]);
    let rhs_layout = layout(&[1, 3]);
    let output_layout = layout(&[2, 3]);
    let lhs = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [2.0_f64, 4.0, 6.0];

    for (operation, expected) in [
        (BinaryOp::Eq, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        (BinaryOp::Ne, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        (BinaryOp::Lt, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        (BinaryOp::Gt, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        (BinaryOp::Le, [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        (BinaryOp::Ge, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    ] {
        let mut actual = [0.0; 6];
        elementwise_binary_into(
            operation,
            &lhs_layout,
            &lhs,
            &rhs_layout,
            &rhs,
            &output_layout,
            &mut actual,
        )
        .expect("broadcast comparison dispatch");
        assert_eq!(actual, expected, "{operation:?}");
    }
}

#[test]
fn binary_dispatch_covers_arithmetic_ops() {
    let la = layout(&[2, 2]);
    let a = vec![8.0f64, 9.0, 10.0, 12.0];
    let b = vec![2.0f64, 3.0, 5.0, 6.0];
    let mut out = vec![0.0f64; 4];

    elementwise_binary_into(BinaryOp::Sub, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![6.0, 6.0, 5.0, 6.0]);

    elementwise_binary_into(BinaryOp::Mul, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![16.0, 27.0, 50.0, 72.0]);

    elementwise_binary_into(BinaryOp::Div, &la, &a, &la, &b, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 3.0, 2.0, 2.0]);
}

#[test]
fn unary_dispatch_covers_scalar_mapping() {
    let input = vec![-4.0f64, -1.0, 0.0, 9.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_unary_into(CpuUnaryOp::Relu, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![0.0, 0.0, 0.0, 9.0]);

    elementwise_unary_into(CpuUnaryOp::Abs, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 1.0, 0.0, 9.0]);

    elementwise_unary_into(CpuUnaryOp::Neg, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![4.0, 1.0, -0.0, -9.0]);
}

#[test]
fn unary_dispatch_exp_log_sqrt_matches_scalar_reference() {
    let input = vec![0.0f64, 1.0, 4.0, 16.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[2, 2]);

    elementwise_unary_into(CpuUnaryOp::Exp, &la, &input, &la, &mut out).unwrap();
    assert_transcendental_close(&out, &[1.0, 1.0_f64.exp(), 4.0_f64.exp(), 16.0_f64.exp()]);

    elementwise_unary_into(CpuUnaryOp::Log, &la, &input, &la, &mut out).unwrap();
    assert_transcendental_close(&out, &[f64::NEG_INFINITY, 0.0, 4.0_f64.ln(), 16.0_f64.ln()]);

    elementwise_unary_into(CpuUnaryOp::Sqrt, &la, &input, &la, &mut out).unwrap();
    assert_eq!(out, vec![0.0, 1.0, 2.0, 4.0]);
}

#[test]
fn unary_dispatch_special_functions_match_reference_values() {
    let input = vec![0.0f64, 0.5, 1.0, 5.0];
    let mut out = vec![0.0f64; 4];
    let la = layout(&[4]);

    elementwise_unary_into(CpuUnaryOp::Erf, &la, &input, &la, &mut out).unwrap();
    assert!(
        (out[1] - 0.520_499_877_813_046_5).abs() <= 2.0e-15,
        "erf(0.5)"
    );

    elementwise_unary_into(CpuUnaryOp::Erfc, &la, &input, &la, &mut out).unwrap();
    assert!(
        (out[3] - 1.537_459_794_428_034_7e-12).abs() <= 2.0e-25,
        "erfc(5)"
    );

    elementwise_unary_into(CpuUnaryOp::Lgamma, &la, &input, &la, &mut out).unwrap();
    assert!(out[0].is_infinite(), "lgamma(0)");
    assert!(
        (out[1] - 0.572_364_942_924_700_1).abs() <= 2.0e-15,
        "lgamma(0.5)"
    );
    assert_eq!(out[2], 0.0);
    assert!((out[3] - 24.0_f64.ln()).abs() <= 2.0e-15, "lgamma(5)");
}
