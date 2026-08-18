//! Finite-difference checks for the element-wise activation derivatives.
//!
//! Each op here supplies its own `backward_from_*` formula through the generic
//! unary node, so every one is a separate hand-derived claim even though they
//! share a node body.
//!
//! # Domains and kinks
//!
//! Two input constraints govern the fixtures:
//!
//! * **Domain.** `log`, `sqrt` and `recip` are undefined or singular at `0`;
//!   `asin`, `acos` and `atanh` need `(-1, 1)`; `acosh` needs `x > 1`. Sampling
//!   outside a domain produces `NaN`, and sampling near a singularity produces a
//!   true derivative so large that no finite-difference bound holds — neither
//!   is a defect in the backward. Each check states the interval it samples.
//!
//! * **Kinks.** The rectifier family is non-differentiable at a known point:
//!   `0` for `relu`/`elu`/`selu`/`celu`/`abs`/`prelu`, `±1` for `hardtanh`,
//!   `±λ` for the shrinkage ops, the threshold for `threshold`, the bounds for
//!   `clamp`. A central difference across such a point averages the two
//!   one-sided slopes and disagrees with any correct backward. The fixtures put
//!   the samples in the interior of the branches; the step is `≈ 6.1e-6`, so an
//!   interval whose endpoints clear the kink by `0.1` cannot be crossed.

use super::{weighted, weighting, Sampler, T64};
use coeus_autograd::{
    abs, acos, acosh, asin, asinh, atan, atanh, celu, clamp, cos, cosh, elu, erf, erfc, exp, exp2,
    expm1, gelu, gelu_tanh, gradcheck, hardshrink, hardsigmoid, hardswish, hardtanh, leaky_relu,
    log, log10, log1p, log2, mish, pow, prelu, recip, relu, selu, sigmoid, silu, sin, sinh,
    softplus, softshrink, softsign, sqrt, tan, tanh, threshold, Var,
};
use coeus_core::MoiraiBackend;

/// Shape every element-wise check uses.
const SHAPE: [usize; 2] = [3, 4];

/// Run `op` on a fixture drawn from `sampler` and require agreement.
///
/// The non-uniform weighting is what makes the reduction non-vacuous; see the
/// module documentation of the parent module.
fn check(sampler: &Sampler, op: impl Fn(&Var<f64, MoiraiBackend>) -> Var<f64, MoiraiBackend>) {
    let x = sampler.tensor(&SHAPE);
    let w = weighting(&SHAPE);
    gradcheck(&[x], |v| weighted(&op(&v[0]), &w))
        .expect("activation backward must match central differences");
}

/// Samples in `(-0.9, 0.9)`, for ops smooth across the origin.
fn smooth(phase: f64) -> Sampler {
    Sampler::signed(phase)
}

/// Samples in `(0.2, 1.8)`, for ops singular or undefined at `0`.
fn positive(phase: f64) -> Sampler {
    Sampler::positive(phase)
}

/// Samples in `(-0.85, 0.85)`, strictly inside the unit interval.
///
/// For `asin`, `acos` and `atanh`, whose derivatives diverge at `±1`; `0.15`
/// of clearance keeps the derivative below `4` and the second derivative
/// bounded, so the `O(1)`-conditioning assumption behind the derived tolerance
/// holds.
fn unit_interior(phase: f64) -> Sampler {
    Sampler::new(phase, -0.85, 0.85)
}

// ── Smooth across the origin ────────────────────────────────────────────────

#[test]
fn sigmoid_backward_matches_finite_differences() {
    check(&smooth(0.11), sigmoid);
}

#[test]
fn tanh_backward_matches_finite_differences() {
    check(&smooth(0.13), tanh);
}

#[test]
fn silu_backward_matches_finite_differences() {
    // x·sigmoid(x): the product rule term is the one an implementation drops.
    check(&smooth(0.17), silu);
}

#[test]
fn mish_backward_matches_finite_differences() {
    // x·tanh(softplus(x)) — a three-deep chain, and the deepest derivative in
    // the activation set.
    check(&smooth(0.19), mish);
}

#[test]
fn softplus_backward_matches_finite_differences() {
    check(&smooth(0.23), softplus);
}

#[test]
fn gelu_backward_matches_finite_differences() {
    // The exact erf form; its derivative carries both the cdf and the pdf term.
    check(&smooth(0.29), gelu);
}

#[test]
fn gelu_tanh_backward_matches_finite_differences() {
    // The tanh approximation is a different function with a different
    // derivative; sharing a backward with the exact form would be a defect.
    check(&smooth(0.31), gelu_tanh);
}

#[test]
fn softsign_backward_matches_finite_differences() {
    // x/(1+|x|): smooth at 0 despite the |x|, since the one-sided slopes agree.
    check(&smooth(0.37), softsign);
}

#[test]
fn exp_backward_matches_finite_differences() {
    check(&smooth(0.41), exp);
}

#[test]
fn exp2_backward_matches_finite_differences() {
    // d/dx 2^x = ln2·2^x; the ln2 factor is the part that gets forgotten.
    check(&smooth(0.43), exp2);
}

#[test]
fn expm1_backward_matches_finite_differences() {
    check(&smooth(0.47), expm1);
}

#[test]
fn sin_backward_matches_finite_differences() {
    check(&smooth(0.53), sin);
}

#[test]
fn cos_backward_matches_finite_differences() {
    // d/dx cos = -sin; the sign is the whole content of the claim.
    check(&smooth(0.59), cos);
}

#[test]
fn tan_backward_matches_finite_differences() {
    // Sampled in (-0.9, 0.9), well inside (-π/2, π/2), so sec²(x) stays below 3.
    check(&smooth(0.61), tan);
}

#[test]
fn sinh_backward_matches_finite_differences() {
    check(&smooth(0.67), sinh);
}

#[test]
fn cosh_backward_matches_finite_differences() {
    check(&smooth(0.71), cosh);
}

#[test]
fn atan_backward_matches_finite_differences() {
    check(&smooth(0.73), atan);
}

#[test]
fn asinh_backward_matches_finite_differences() {
    check(&smooth(0.79), asinh);
}

#[test]
fn erf_backward_matches_finite_differences() {
    check(&smooth(0.83), erf);
}

#[test]
fn erfc_backward_matches_finite_differences() {
    // erfc' = -erf'; a shared implementation that forgot the sign fails here
    // and nowhere else.
    check(&smooth(0.89), erfc);
}

#[test]
fn log1p_backward_matches_finite_differences() {
    // Domain x > -1; the sampler's lower bound of -0.9 clears it by 0.1.
    check(&smooth(0.97), log1p);
}

// ── Positive domain ─────────────────────────────────────────────────────────

#[test]
fn log_backward_matches_finite_differences() {
    check(&positive(0.12), log);
}

#[test]
fn log2_backward_matches_finite_differences() {
    // d/dx log2(x) = 1/(x·ln2).
    check(&positive(0.14), log2);
}

#[test]
fn log10_backward_matches_finite_differences() {
    check(&positive(0.18), log10);
}

#[test]
fn sqrt_backward_matches_finite_differences() {
    // Singular at 0; sampled from 0.2 up, where 1/(2√x) stays below 1.2.
    check(&positive(0.22), sqrt);
}

#[test]
fn recip_backward_matches_finite_differences() {
    // -1/x²; at the sampler's lower bound of 0.2 that is -25, large but finite
    // and well within the relative term of the derived tolerance.
    check(&positive(0.26), recip);
}

#[test]
fn pow_backward_matches_finite_differences() {
    // A non-integer exponent needs a strictly positive base, and exercises the
    // general p·x^(p-1) rule rather than an integer special case.
    check(&positive(0.32), |x| pow(x, 2.5));
}

#[test]
fn acosh_backward_matches_finite_differences() {
    // Domain x > 1, singular at 1; sampled from 1.3 up.
    check(&Sampler::new(0.34, 1.3, 2.9), acosh);
}

// ── Bounded domain ──────────────────────────────────────────────────────────

#[test]
fn asin_backward_matches_finite_differences() {
    check(&unit_interior(0.38), asin);
}

#[test]
fn acos_backward_matches_finite_differences() {
    // acos' = -asin'; again the sign is the claim.
    check(&unit_interior(0.42), acos);
}

#[test]
fn atanh_backward_matches_finite_differences() {
    check(&unit_interior(0.46), atanh);
}

// ── Kinked: samples confined to one branch's interior ───────────────────────

#[test]
fn relu_backward_matches_finite_differences_on_the_positive_branch() {
    // Sampled in (0.2, 1.8): the derivative is identically 1 and no sample can
    // reach the kink at 0. The negative branch is covered below.
    check(&positive(0.52), relu);
}

#[test]
fn relu_backward_is_zero_on_the_negative_branch() {
    // Entirely negative input: relu is constant, so both the analytic and the
    // numeric gradient are zero and `gradcheck` would correctly reject the
    // comparison as vacuous. The claim is therefore asserted directly — the
    // gradient must be exactly zero, not merely small.
    let x = Sampler::new(0.54, -1.8, -0.2).tensor(&SHAPE);
    let tracked = Var::new(x, true);
    let w = weighting(&SHAPE);
    weighted(&relu(&tracked), &w)
        .backward()
        .expect("relu backward completes");
    let grad = tracked.grad().expect("input must receive a gradient");
    for (index, &component) in grad.as_slice().iter().enumerate() {
        assert_eq!(
            component, 0.0,
            "relu gradient at element {index} must be exactly zero below the kink"
        );
    }
}

#[test]
fn leaky_relu_backward_matches_finite_differences_below_the_kink() {
    // The negative branch is where the slope parameter lives, so it is the
    // branch worth checking: a leaky_relu that ignored `negative_slope` is
    // indistinguishable from relu above zero.
    check(&Sampler::new(0.56, -1.8, -0.2), |x| leaky_relu(x, 0.125));
}

#[test]
fn elu_backward_matches_finite_differences_below_the_kink() {
    // exp(x) - 1 on the negative branch; the derivative exp(x) is the
    // non-trivial half of the rule.
    check(&Sampler::new(0.58, -1.8, -0.2), elu);
}

#[test]
fn selu_backward_matches_finite_differences_below_the_kink() {
    // SELU's negative branch carries both the fixed alpha and the outer lambda;
    // dropping either leaves a gradient wrong by a constant factor.
    check(&Sampler::new(0.62, -1.8, -0.2), selu);
}

#[test]
fn celu_backward_matches_finite_differences_below_the_kink() {
    // alpha appears twice — inside the exponent and as the outer scale — and
    // the two uses cancel in the derivative only if both are present.
    check(&Sampler::new(0.64, -1.8, -0.2), |x| celu(x, 1.5));
}

#[test]
fn abs_backward_matches_finite_differences_on_the_negative_branch() {
    // d|x|/dx = -1 for x < 0; sampled entirely below zero so the kink is not
    // straddled.
    check(&Sampler::new(0.68, -1.8, -0.2), abs);
}

#[test]
fn hardtanh_backward_matches_finite_differences_in_the_linear_region() {
    // Bounds at ±1, samples in (-0.85, 0.85): inside the identity region, where
    // the derivative is 1. Outside it the function is constant and the
    // comparison would be vacuous.
    check(&unit_interior(0.72), |x| hardtanh(x, -1.0, 1.0));
}

#[test]
fn hardsigmoid_backward_matches_finite_differences_in_the_linear_region() {
    // Breakpoints at ±3; samples stay within ±0.9 of the origin, deep inside
    // the sloped region where the derivative is the constant 1/6.
    check(&smooth(0.74), hardsigmoid);
}

#[test]
fn hardswish_backward_matches_finite_differences_in_the_quadratic_region() {
    // x·hardsigmoid(x) is quadratic between the breakpoints, so unlike
    // hardsigmoid its derivative varies with x and a constant-slope error is
    // detectable.
    check(&smooth(0.76), hardswish);
}

#[test]
fn hardshrink_backward_matches_finite_differences_above_the_threshold() {
    // lambda = 0.5, samples in (0.7, 2.1): clear of the kink at 0.5 by 0.2,
    // in the region where the derivative is 1.
    check(&Sampler::new(0.78, 0.7, 2.1), |x| hardshrink(x, 0.5));
}

#[test]
fn softshrink_backward_matches_finite_differences_above_the_threshold() {
    // Same placement as hardshrink; softshrink subtracts lambda rather than
    // passing x through, but the derivative is 1 in the same region, so a
    // backward that confused the two is caught by value elsewhere, not here.
    check(&Sampler::new(0.82, 0.7, 2.1), |x| softshrink(x, 0.5));
}

#[test]
fn threshold_backward_matches_finite_differences_above_the_threshold() {
    // threshold = 0.5, replacement value 0.1; samples in (0.7, 2.1) pass
    // through unchanged with derivative 1.
    check(&Sampler::new(0.86, 0.7, 2.1), |x| threshold(x, 0.5, 0.1));
}

#[test]
fn clamp_backward_matches_finite_differences_inside_the_bounds() {
    // Bounds at ±1.2, samples within ±0.9: strictly interior, so the derivative
    // is 1 and the saturated branches — where it is 0 — do not make the
    // comparison vacuous.
    check(&smooth(0.88), |x| clamp(x, -1.2, 1.2));
}

#[test]
fn prelu_backward_matches_finite_differences_below_the_kink() {
    // The learned slope is differentiated alongside the input: dL/da = Σ x·dy
    // over the negative entries only, a rule distinct from the input's and one
    // that a shared implementation gets wrong by summing over all entries.
    let x = Sampler::new(0.92, -1.8, -0.2).tensor(&SHAPE);
    let slope = T64::from_slice_on([1], &[0.25], &MoiraiBackend::new());
    let w = weighting(&SHAPE);

    gradcheck(&[x, slope], |v| weighted(&prelu(&v[0], &v[1]), &w))
        .expect("prelu backward must match central differences");
}
