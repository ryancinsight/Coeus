//! Finite-difference checks for the elementary element-wise functions.

use super::{check, positive, smooth, unit_interior, GradcheckScalar, Sampler};
use coeus_autograd::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, erf, erfc, exp, exp2, expm1, log, log10,
    log1p, log2, pow, recip, sin, sinh, sqrt, tan, Var,
};
use coeus_core::MoiraiBackend;

fn exp_case<T: GradcheckScalar>() {
    check(&smooth(0.41), exp::<T, MoiraiBackend>);
}

#[test]
fn exp_backward_matches_finite_differences() {
    exp_case::<f64>();
    exp_case::<f32>();
}

fn exp2_case<T: GradcheckScalar>() {
    // d/dx 2^x = ln2·2^x; the ln2 factor is the part that gets forgotten.
    check(&smooth(0.43), exp2::<T, MoiraiBackend>);
}

#[test]
fn exp2_backward_matches_finite_differences() {
    exp2_case::<f64>();
    exp2_case::<f32>();
}

fn expm1_case<T: GradcheckScalar>() {
    check(&smooth(0.47), expm1::<T, MoiraiBackend>);
}

#[test]
fn expm1_backward_matches_finite_differences() {
    expm1_case::<f64>();
    expm1_case::<f32>();
}

fn sin_case<T: GradcheckScalar>() {
    check(&smooth(0.53), sin::<T, MoiraiBackend>);
}

#[test]
fn sin_backward_matches_finite_differences() {
    sin_case::<f64>();
    sin_case::<f32>();
}

fn cos_case<T: GradcheckScalar>() {
    // d/dx cos = -sin; the sign is the whole content of the claim.
    check(&smooth(0.59), cos::<T, MoiraiBackend>);
}

#[test]
fn cos_backward_matches_finite_differences() {
    cos_case::<f64>();
    cos_case::<f32>();
}

fn tan_case<T: GradcheckScalar>() {
    // Sampled in (-0.9, 0.9), well inside (-π/2, π/2), so sec²(x) stays below 3.
    check(&smooth(0.61), tan::<T, MoiraiBackend>);
}

#[test]
fn tan_backward_matches_finite_differences() {
    tan_case::<f64>();
    tan_case::<f32>();
}

fn sinh_case<T: GradcheckScalar>() {
    check(&smooth(0.67), sinh::<T, MoiraiBackend>);
}

#[test]
fn sinh_backward_matches_finite_differences() {
    sinh_case::<f64>();
    sinh_case::<f32>();
}

fn cosh_case<T: GradcheckScalar>() {
    check(&smooth(0.71), cosh::<T, MoiraiBackend>);
}

#[test]
fn cosh_backward_matches_finite_differences() {
    cosh_case::<f64>();
    cosh_case::<f32>();
}

fn atan_case<T: GradcheckScalar>() {
    check(&smooth(0.73), atan::<T, MoiraiBackend>);
}

#[test]
fn atan_backward_matches_finite_differences() {
    atan_case::<f64>();
    atan_case::<f32>();
}

fn asinh_case<T: GradcheckScalar>() {
    check(&smooth(0.79), asinh::<T, MoiraiBackend>);
}

#[test]
fn asinh_backward_matches_finite_differences() {
    asinh_case::<f64>();
    asinh_case::<f32>();
}

fn erf_case<T: GradcheckScalar>() {
    check(&smooth(0.83), erf::<T, MoiraiBackend>);
}

#[test]
fn erf_backward_matches_finite_differences() {
    erf_case::<f64>();
    erf_case::<f32>();
}

fn erfc_case<T: GradcheckScalar>() {
    // erfc' = -erf'; a shared implementation that forgot the sign fails here
    // and nowhere else.
    check(&smooth(0.89), erfc::<T, MoiraiBackend>);
}

#[test]
fn erfc_backward_matches_finite_differences() {
    erfc_case::<f64>();
    erfc_case::<f32>();
}

fn log1p_case<T: GradcheckScalar>() {
    // Domain x > -1; the sampler's lower bound of -0.9 clears it by 0.1.
    check(&smooth(0.97), log1p::<T, MoiraiBackend>);
}

#[test]
fn log1p_backward_matches_finite_differences() {
    log1p_case::<f64>();
    log1p_case::<f32>();
}

fn log_case<T: GradcheckScalar>() {
    check(&positive(0.12), log::<T, MoiraiBackend>);
}

#[test]
fn log_backward_matches_finite_differences() {
    log_case::<f64>();
    log_case::<f32>();
}

fn log2_case<T: GradcheckScalar>() {
    // d/dx log2(x) = 1/(x·ln2).
    check(&positive(0.14), log2::<T, MoiraiBackend>);
}

#[test]
fn log2_backward_matches_finite_differences() {
    log2_case::<f64>();
    log2_case::<f32>();
}

fn log10_case<T: GradcheckScalar>() {
    check(&positive(0.18), log10::<T, MoiraiBackend>);
}

#[test]
fn log10_backward_matches_finite_differences() {
    log10_case::<f64>();
    log10_case::<f32>();
}

fn sqrt_case<T: GradcheckScalar>() {
    // Singular at 0; sampled from 0.2 up, where 1/(2√x) stays below 1.2.
    check(&positive(0.22), sqrt::<T, MoiraiBackend>);
}

#[test]
fn sqrt_backward_matches_finite_differences() {
    sqrt_case::<f64>();
    sqrt_case::<f32>();
}

fn recip_case<T: GradcheckScalar>() {
    // -1/x²; at the sampler's lower bound of 0.2 that is -25, large but finite
    // and well within the relative term of the derived tolerance.
    check(&positive(0.26), recip::<T, MoiraiBackend>);
}

#[test]
fn recip_backward_matches_finite_differences() {
    recip_case::<f64>();
    recip_case::<f32>();
}

fn pow_case<T: GradcheckScalar>() {
    // A non-integer exponent needs a strictly positive base, and exercises the
    // general p·x^(p-1) rule rather than an integer special case.
    check(&positive(0.32), |x: &Var<T, MoiraiBackend>| pow(x, 2.5));
}

#[test]
fn pow_backward_matches_finite_differences() {
    pow_case::<f64>();
    pow_case::<f32>();
}

fn acosh_case<T: GradcheckScalar>() {
    // Domain x > 1, singular at 1; sampled from 1.3 up.
    check(&Sampler::new(0.34, 1.3, 2.9), acosh::<T, MoiraiBackend>);
}

#[test]
fn acosh_backward_matches_finite_differences() {
    acosh_case::<f64>();
    acosh_case::<f32>();
}

fn asin_case<T: GradcheckScalar>() {
    check(&unit_interior(0.38), asin::<T, MoiraiBackend>);
}

#[test]
fn asin_backward_matches_finite_differences() {
    asin_case::<f64>();
    asin_case::<f32>();
}

fn acos_case<T: GradcheckScalar>() {
    // acos' = -asin'; again the sign is the claim.
    check(&unit_interior(0.42), acos::<T, MoiraiBackend>);
}

#[test]
fn acos_backward_matches_finite_differences() {
    acos_case::<f64>();
    acos_case::<f32>();
}

fn atanh_case<T: GradcheckScalar>() {
    check(&unit_interior(0.46), atanh::<T, MoiraiBackend>);
}

#[test]
fn atanh_backward_matches_finite_differences() {
    atanh_case::<f64>();
    atanh_case::<f32>();
}
