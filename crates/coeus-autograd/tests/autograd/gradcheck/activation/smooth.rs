//! Finite-difference checks for the activations that are smooth across the origin.

use super::{check, smooth, GradcheckScalar};
use coeus_autograd::{gelu, gelu_tanh, mish, sigmoid, silu, softplus, softsign, tanh};
use coeus_core::MoiraiBackend;

fn sigmoid_case<T: GradcheckScalar>() {
    check(&smooth(0.11), sigmoid::<T, MoiraiBackend>);
}

#[test]
fn sigmoid_backward_matches_finite_differences() {
    sigmoid_case::<f64>();
    sigmoid_case::<f32>();
}

fn tanh_case<T: GradcheckScalar>() {
    check(&smooth(0.13), tanh::<T, MoiraiBackend>);
}

#[test]
fn tanh_backward_matches_finite_differences() {
    tanh_case::<f64>();
    tanh_case::<f32>();
}

fn silu_case<T: GradcheckScalar>() {
    // x·sigmoid(x): the product rule term is the one an implementation drops.
    check(&smooth(0.17), silu::<T, MoiraiBackend>);
}

#[test]
fn silu_backward_matches_finite_differences() {
    silu_case::<f64>();
    silu_case::<f32>();
}

fn mish_case<T: GradcheckScalar>() {
    // x·tanh(softplus(x)) — a three-deep chain, and the deepest derivative in
    // the activation set.
    check(&smooth(0.19), mish::<T, MoiraiBackend>);
}

#[test]
fn mish_backward_matches_finite_differences() {
    mish_case::<f64>();
    mish_case::<f32>();
}

fn softplus_case<T: GradcheckScalar>() {
    check(&smooth(0.23), softplus::<T, MoiraiBackend>);
}

#[test]
fn softplus_backward_matches_finite_differences() {
    softplus_case::<f64>();
    softplus_case::<f32>();
}

fn gelu_case<T: GradcheckScalar>() {
    // The exact erf form; its derivative carries both the cdf and the pdf term.
    check(&smooth(0.29), gelu::<T, MoiraiBackend>);
}

#[test]
fn gelu_backward_matches_finite_differences() {
    gelu_case::<f64>();
    gelu_case::<f32>();
}

fn gelu_tanh_case<T: GradcheckScalar>() {
    // The tanh approximation is a different function with a different
    // derivative; sharing a backward with the exact form would be a defect.
    check(&smooth(0.31), gelu_tanh::<T, MoiraiBackend>);
}

#[test]
fn gelu_tanh_backward_matches_finite_differences() {
    gelu_tanh_case::<f64>();
    gelu_tanh_case::<f32>();
}

fn softsign_case<T: GradcheckScalar>() {
    // x/(1+|x|): smooth at 0 despite the |x|, since the one-sided slopes agree.
    check(&smooth(0.37), softsign::<T, MoiraiBackend>);
}

#[test]
fn softsign_backward_matches_finite_differences() {
    softsign_case::<f64>();
    softsign_case::<f32>();
}
