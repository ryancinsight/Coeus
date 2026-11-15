//! Finite-difference checks for the kinked rectifier family.
//!
//! Every fixture keeps its samples inside one branch's interior, because a
//! central difference across the kink averages the two one-sided slopes.

use super::{
    check, positive, smooth, unit_interior, weighted, weighting, GradcheckScalar, Sampler, Var,
    SHAPE,
};
use coeus_autograd::{
    abs, celu, clamp, elu, gradcheck, hardshrink, hardsigmoid, hardswish, hardtanh, leaky_relu,
    prelu, relu, selu, softshrink, threshold,
};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn relu_positive_branch_case<T: GradcheckScalar>() {
    // Sampled in (0.2, 1.8): the derivative is identically 1 and no sample can
    // reach the kink at 0. The negative branch is covered below.
    check(&positive(0.52), relu::<T, MoiraiBackend>);
}

#[test]
fn relu_backward_matches_finite_differences_on_the_positive_branch() {
    relu_positive_branch_case::<f64>();
    relu_positive_branch_case::<f32>();
}

fn relu_negative_branch_zero_case<T: GradcheckScalar>() {
    // Entirely negative input: relu is constant, so both the analytic and the
    // numeric gradient are zero and `gradcheck` would correctly reject the
    // comparison as vacuous. The claim is therefore asserted directly — the
    // gradient must be exactly zero, not merely small.
    let x = Sampler::new(0.54, -1.8, -0.2).tensor::<T>(&SHAPE);
    let tracked = Var::new(x, true).expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&SHAPE);
    weighted(
        &relu(&tracked).expect("invariant: test operation succeeds"),
        &w,
    )
    .backward()
    .expect("relu backward completes");
    let grad = tracked.grad().expect("input must receive a gradient");
    let zero = <T as coeus_core::FloatElement>::from_f64(0.0);
    for (index, &component) in grad.as_slice().iter().enumerate() {
        assert_eq!(
            component, zero,
            "relu gradient at element {index} must be exactly zero below the kink"
        );
    }
}

#[test]
fn relu_backward_is_zero_on_the_negative_branch() {
    relu_negative_branch_zero_case::<f64>();
    relu_negative_branch_zero_case::<f32>();
}

fn leaky_relu_below_kink_case<T: GradcheckScalar>() {
    // The negative branch is where the slope parameter lives, so it is the
    // branch worth checking: a leaky_relu that ignored `negative_slope` is
    // indistinguishable from relu above zero.
    check(
        &Sampler::new(0.56, -1.8, -0.2),
        |x: &Var<T, MoiraiBackend>| leaky_relu(x, 0.125),
    );
}

#[test]
fn leaky_relu_backward_matches_finite_differences_below_the_kink() {
    leaky_relu_below_kink_case::<f64>();
    leaky_relu_below_kink_case::<f32>();
}

fn elu_below_kink_case<T: GradcheckScalar>() {
    // exp(x) - 1 on the negative branch; the derivative exp(x) is the
    // non-trivial half of the rule.
    check(&Sampler::new(0.58, -1.8, -0.2), elu::<T, MoiraiBackend>);
}

#[test]
fn elu_backward_matches_finite_differences_below_the_kink() {
    elu_below_kink_case::<f64>();
    elu_below_kink_case::<f32>();
}

fn selu_below_kink_case<T: GradcheckScalar>() {
    // SELU's negative branch carries both the fixed alpha and the outer lambda;
    // dropping either leaves a gradient wrong by a constant factor.
    check(&Sampler::new(0.62, -1.8, -0.2), selu::<T, MoiraiBackend>);
}

#[test]
fn selu_backward_matches_finite_differences_below_the_kink() {
    selu_below_kink_case::<f64>();
    selu_below_kink_case::<f32>();
}

fn celu_below_kink_case<T: GradcheckScalar>() {
    // alpha appears twice — inside the exponent and as the outer scale — and
    // the two uses cancel in the derivative only if both are present.
    check(
        &Sampler::new(0.64, -1.8, -0.2),
        |x: &Var<T, MoiraiBackend>| celu(x, 1.5),
    );
}

#[test]
fn celu_backward_matches_finite_differences_below_the_kink() {
    celu_below_kink_case::<f64>();
    celu_below_kink_case::<f32>();
}

fn abs_negative_branch_case<T: GradcheckScalar>() {
    // d|x|/dx = -1 for x < 0; sampled entirely below zero so the kink is not
    // straddled.
    check(&Sampler::new(0.68, -1.8, -0.2), abs::<T, MoiraiBackend>);
}

#[test]
fn abs_backward_matches_finite_differences_on_the_negative_branch() {
    abs_negative_branch_case::<f64>();
    abs_negative_branch_case::<f32>();
}

fn hardtanh_linear_region_case<T: GradcheckScalar>() {
    // Bounds at ±1, samples in (-0.85, 0.85): inside the identity region, where
    // the derivative is 1. Outside it the function is constant and the
    // comparison would be vacuous.
    check(&unit_interior(0.72), |x: &Var<T, MoiraiBackend>| {
        hardtanh(x, -1.0, 1.0)
    });
}

#[test]
fn hardtanh_backward_matches_finite_differences_in_the_linear_region() {
    hardtanh_linear_region_case::<f64>();
    hardtanh_linear_region_case::<f32>();
}

fn hardsigmoid_linear_region_case<T: GradcheckScalar>() {
    // Breakpoints at ±3; samples stay within ±0.9 of the origin, deep inside
    // the sloped region where the derivative is the constant 1/6.
    check(&smooth(0.74), hardsigmoid::<T, MoiraiBackend>);
}

#[test]
fn hardsigmoid_backward_matches_finite_differences_in_the_linear_region() {
    hardsigmoid_linear_region_case::<f64>();
    hardsigmoid_linear_region_case::<f32>();
}

fn hardswish_quadratic_region_case<T: GradcheckScalar>() {
    // x·hardsigmoid(x) is quadratic between the breakpoints, so unlike
    // hardsigmoid its derivative varies with x and a constant-slope error is
    // detectable.
    check(&smooth(0.76), hardswish::<T, MoiraiBackend>);
}

#[test]
fn hardswish_backward_matches_finite_differences_in_the_quadratic_region() {
    hardswish_quadratic_region_case::<f64>();
    hardswish_quadratic_region_case::<f32>();
}

fn hardshrink_above_threshold_case<T: GradcheckScalar>() {
    // lambda = 0.5, samples in (0.7, 2.1): clear of the kink at 0.5 by 0.2,
    // in the region where the derivative is 1.
    check(
        &Sampler::new(0.78, 0.7, 2.1),
        |x: &Var<T, MoiraiBackend>| hardshrink(x, 0.5),
    );
}

#[test]
fn hardshrink_backward_matches_finite_differences_above_the_threshold() {
    hardshrink_above_threshold_case::<f64>();
    hardshrink_above_threshold_case::<f32>();
}

fn softshrink_above_threshold_case<T: GradcheckScalar>() {
    // Same placement as hardshrink; softshrink subtracts lambda rather than
    // passing x through, but the derivative is 1 in the same region, so a
    // backward that confused the two is caught by value elsewhere, not here.
    check(
        &Sampler::new(0.82, 0.7, 2.1),
        |x: &Var<T, MoiraiBackend>| softshrink(x, 0.5),
    );
}

#[test]
fn softshrink_backward_matches_finite_differences_above_the_threshold() {
    softshrink_above_threshold_case::<f64>();
    softshrink_above_threshold_case::<f32>();
}

fn threshold_above_threshold_case<T: GradcheckScalar>() {
    // threshold = 0.5, replacement value 0.1; samples in (0.7, 2.1) pass
    // through unchanged with derivative 1.
    check(
        &Sampler::new(0.86, 0.7, 2.1),
        |x: &Var<T, MoiraiBackend>| threshold(x, 0.5, 0.1),
    );
}

#[test]
fn threshold_backward_matches_finite_differences_above_the_threshold() {
    threshold_above_threshold_case::<f64>();
    threshold_above_threshold_case::<f32>();
}

fn clamp_inside_bounds_case<T: GradcheckScalar>() {
    // Bounds at ±1.2, samples within ±0.9: strictly interior, so the derivative
    // is 1 and the saturated branches — where it is 0 — do not make the
    // comparison vacuous.
    check(&smooth(0.88), |x: &Var<T, MoiraiBackend>| {
        clamp(
            x,
            <T as coeus_core::FloatElement>::from_f64(-1.2),
            <T as coeus_core::FloatElement>::from_f64(1.2),
        )
    });
}

#[test]
fn clamp_backward_matches_finite_differences_inside_the_bounds() {
    clamp_inside_bounds_case::<f64>();
    clamp_inside_bounds_case::<f32>();
}

fn prelu_below_kink_case<T: GradcheckScalar>() {
    // The learned slope is differentiated alongside the input: dL/da = Σ x·dy
    // over the negative entries only, a rule distinct from the input's and one
    // that a shared implementation gets wrong by summing over all entries.
    let x = Sampler::new(0.92, -1.8, -0.2).tensor::<T>(&SHAPE);
    let slope = Tensor::<T, MoiraiBackend>::from_slice_on(
        [1],
        &[<T as coeus_core::FloatElement>::from_f64(0.25)],
        &MoiraiBackend::new(),
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&SHAPE);

    gradcheck(&[x, slope], |v| {
        weighted(
            &prelu(&v[0], &v[1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("prelu backward must match central differences");
}

#[test]
fn prelu_backward_matches_finite_differences_below_the_kink() {
    prelu_below_kink_case::<f64>();
    prelu_below_kink_case::<f32>();
}
