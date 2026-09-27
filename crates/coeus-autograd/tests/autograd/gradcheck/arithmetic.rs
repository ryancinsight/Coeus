//! Finite-difference checks for the elementwise binary/unary arithmetic ops
//! that had no gradcheck coverage: `add`, `sub`, `div`, `remainder`,
//! `maximum`, `minimum`, `neg`.
//!
//! Each check is one generic function instantiated at every scalar type this
//! module verifies (`f64` for oracle sensitivity, `f32` for instantiation
//! coverage — see the module documentation), called from a single `#[test]`
//! so nextest reports one falsifiable property per op rather than one test
//! per type.

use super::{tensor, weighted, weighting, GradcheckScalar, Sampler};
use coeus_autograd::{add, div, gradcheck, maximum, minimum, neg, remainder, sub};

fn add_case<T: GradcheckScalar>() {
    let a = tensor::<T>(&[3, 4], 0.19);
    let b = tensor::<T>(&[3, 4], 0.53);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&add(&v[0], &v[1]), &w))
        .expect("add backward must match central differences");
}

#[test]
fn add_backward_matches_finite_differences() {
    add_case::<f64>();
    add_case::<f32>();
}

fn sub_case<T: GradcheckScalar>() {
    let a = tensor::<T>(&[3, 4], 0.23);
    let b = tensor::<T>(&[3, 4], 0.61);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&sub(&v[0], &v[1]), &w))
        .expect("sub backward must match central differences");
}

#[test]
fn sub_backward_matches_finite_differences() {
    sub_case::<f64>();
    sub_case::<f32>();
}

fn div_case<T: GradcheckScalar>() {
    // The denominator stays in `Sampler::positive`'s (0.2, 1.8), away from the
    // zero the quotient's derivative diverges at.
    let a = tensor::<T>(&[3, 4], 0.29);
    let b = Sampler::positive(0.67).tensor::<T>(&[3, 4]);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&div(&v[0], &v[1]), &w))
        .expect("div backward must match central differences");
}

#[test]
fn div_backward_matches_finite_differences() {
    div_case::<f64>();
    div_case::<f32>();
}

fn remainder_case<T: GradcheckScalar>() {
    // `∂/∂b = -floor(a/b)` (`binary.rs::RemainderOp::backward`) has a jump
    // discontinuity at *every* integer value of `a/b`, unlike a single-kink op
    // such as `relu`. `a ∈ (0.2, 1.8)` and `b ∈ (3.0, 5.0)` bound `a/b` to
    // `(0.04, 0.6)`, strictly inside the one floor-interval `[0, 1)` with
    // margin far wider than the finite-difference step at either scalar
    // width, so no sampled element or its perturbed neighbours can cross a
    // discontinuity. (An earlier signed/positive fixture crossed `a/b == 0`
    // and floor-unit boundaries and produced a large, correctly-rejected
    // mismatch — the fixture was invalid, not the backward formula.)
    let a = Sampler::positive(0.31).tensor::<T>(&[3, 4]);
    let b = Sampler::new(0.73, 3.0, 5.0).tensor::<T>(&[3, 4]);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&remainder(&v[0], &v[1]), &w))
        .expect("remainder backward must match central differences");
}

#[test]
fn remainder_backward_matches_finite_differences() {
    remainder_case::<f64>();
    remainder_case::<f32>();
}

fn maximum_case<T: GradcheckScalar>() {
    // `maximum`'s gradient is a hard switch at `a == b`; the irrational
    // `Sampler` sequence gives two independently-phased fixtures that are
    // never equal at any sampled element.
    let a = tensor::<T>(&[3, 4], 0.37);
    let b = tensor::<T>(&[3, 4], 0.79);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&maximum(&v[0], &v[1]), &w))
        .expect("maximum backward must match central differences");
}

#[test]
fn maximum_backward_matches_finite_differences() {
    maximum_case::<f64>();
    maximum_case::<f32>();
}

fn minimum_case<T: GradcheckScalar>() {
    let a = tensor::<T>(&[3, 4], 0.41);
    let b = tensor::<T>(&[3, 4], 0.83);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a, b], |v| weighted(&minimum(&v[0], &v[1]), &w))
        .expect("minimum backward must match central differences");
}

#[test]
fn minimum_backward_matches_finite_differences() {
    minimum_case::<f64>();
    minimum_case::<f32>();
}

fn neg_case<T: GradcheckScalar>() {
    let a = tensor::<T>(&[3, 4], 0.43);
    let w = weighting::<T>(&[3, 4]);

    gradcheck(&[a], |v| weighted(&neg(&v[0]), &w))
        .expect("neg backward must match central differences");
}

#[test]
fn neg_backward_matches_finite_differences() {
    neg_case::<f64>();
    neg_case::<f32>();
}
