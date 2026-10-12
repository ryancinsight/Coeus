//! Gradcheck self-tests.

use super::*;
use crate::ops::{mul, softmax, sum};
use coeus_core::MoiraiBackend;

fn vector(values: &[f64]) -> Tensor<f64, MoiraiBackend> {
    Tensor::from_slice_on([values.len()], values, &MoiraiBackend::new())
}

fn weights(values: &[f64]) -> Var<f64, MoiraiBackend> {
    Var::new(vector(values), false)
}

#[test]
fn accepts_a_correct_gradient() {
    let w = weights(&[1.0, -2.0, 0.5]);
    gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| sum(&mul(&v[0], &w)))
        .expect("d/dx sum(w·x) = w is exact");
}

#[test]
fn rejects_a_vacuous_all_zero_comparison() {
    // sum(softmax(x)) is identically 1, so every gradient component is
    // exactly zero and the comparison discriminates nothing.
    let error = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| sum(&softmax(&v[0], 0)))
        .expect_err("a zero-vs-zero comparison must be rejected");
    assert!(
        matches!(error, GradcheckError::TriviallyZero { .. }),
        "expected TriviallyZero, got {error:?}"
    );
}

#[test]
fn rejects_a_non_scalar_loss() {
    let w = weights(&[1.0, -2.0, 0.5]);
    let error = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| mul(&v[0], &w))
        .expect_err("a vector loss must be rejected");
    assert!(
        matches!(error, GradcheckError::NonScalarLoss { ref shape } if shape == &[3]),
        "expected NonScalarLoss([3]), got {error:?}"
    );
}

#[test]
fn detects_a_wrong_gradient() {
    // The oracle must be able to fail: compare the gradient of sum(w·x)
    // against a forward whose weighting differs, and the mismatch must
    // surface rather than be absorbed by the tolerance.
    let truthful = weights(&[1.0, -2.0, 0.5]);
    let analytic_only = gradcheck(&[vector(&[0.5, -1.25, 2.0])], |v| {
        // A closure that is *not* a pure function of `v` alone: the tracked
        // call and the perturbed calls disagree, which is exactly the shape
        // of an implementation/derivation divergence.
        if v[0].grad.is_some() {
            sum(&mul(&v[0], &truthful))
        } else {
            sum(&mul(&v[0], &weights(&[1.0, -2.0, 1.5])))
        }
    });
    let error = analytic_only.expect_err("a divergent forward must be detected");
    assert!(
        matches!(error, GradcheckError::Mismatch { .. }),
        "expected Mismatch, got {error:?}"
    );
}

#[test]
fn step_and_floor_follow_machine_epsilon() {
    // The derived epsilon must equal the IEEE constant exactly, for every
    // scalar the check supports — this is the independent oracle for
    // `machine_epsilon`, which cannot read those constants itself.
    assert_eq!(machine_epsilon::<f64>(), f64::EPSILON);
    assert_eq!(machine_epsilon::<f32>(), f64::from(f32::EPSILON));

    // The step is ε^(1/3) and the accuracy floor ε^(2/3); these are the
    // numbers the module doc tabulates, asserted so the derivation and the
    // code cannot drift apart.
    let eps64 = f64::EPSILON;
    assert!((eps64.cbrt() - 6.055e-6).abs() < 1e-9, "f64 step {eps64:e}");
    let floor64 = eps64.cbrt() * eps64.cbrt();
    assert!((floor64 - 3.666e-11).abs() < 1e-14, "f64 floor {floor64:e}");

    let eps32 = f64::from(f32::EPSILON);
    assert!((eps32.cbrt() - 4.921e-3).abs() < 1e-6, "f32 step {eps32:e}");
}
