//! Finite-difference checks for max/avg pooling, tested through their
//! `coeus-nn` `Module` layers (a dev-dependency of this crate) rather than by
//! reproducing `checked_out_dim`/backend-call plumbing here: pooling's output
//! shape depends on kernel/stride/padding/dilation, and duplicating that
//! arithmetic would put the same computation on both sides of the comparison
//! for the output-shape half of the contract. The layer's `forward` is the
//! independent oracle's forward function; the backward under test is still
//! `coeus-autograd`'s `{max,avg}_pool{1,2,3}d`, which the layer calls
//! internally to attach the tracked node.
//!
//! Max pooling has a kink wherever two pooled elements tie for the maximum
//! (the sub-gradient is not unique there); the irrational `Sampler` sequence
//! never produces an exact tie. Average pooling is linear in its input, so it
//! has no kink for any fixture.

use super::{tensor, weighted, weighting, GradcheckScalar};
use coeus_autograd::gradcheck;
use coeus_nn::{AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d, Module};

fn max_pool1d_case<T: GradcheckScalar>() {
    // [N=2, C=2, L=6], kernel 3 stride 1: output L_out = 4.
    let x = tensor::<T>(&[2, 2, 6], 0.17);
    let w = weighting::<T>(&[2, 2, 4]);
    let layer = MaxPool1d::<T>::with_params(3, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("max_pool1d backward must match central differences");
}

#[test]
fn max_pool1d_backward_matches_finite_differences() {
    max_pool1d_case::<f64>();
    max_pool1d_case::<f32>();
}

fn max_pool2d_case<T: GradcheckScalar>() {
    // [N=1, C=2, H=5, W=5], kernel 2 stride 2: output [1,2,2,2].
    let x = tensor::<T>(&[1, 2, 5, 5], 0.23);
    let w = weighting::<T>(&[1, 2, 2, 2]);
    let layer = MaxPool2d::<T>::with_params(2, 2, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("max_pool2d backward must match central differences");
}

#[test]
fn max_pool2d_backward_matches_finite_differences() {
    max_pool2d_case::<f64>();
    max_pool2d_case::<f32>();
}

fn max_pool3d_case<T: GradcheckScalar>() {
    // [N=1, C=1, D=4, H=4, W=4], kernel 2 stride 2: output [1,1,2,2,2].
    let x = tensor::<T>(&[1, 1, 4, 4, 4], 0.29);
    let w = weighting::<T>(&[1, 1, 2, 2, 2]);
    let layer = MaxPool3d::<T>::with_params(2, 2, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("max_pool3d backward must match central differences");
}

#[test]
fn max_pool3d_backward_matches_finite_differences() {
    max_pool3d_case::<f64>();
    max_pool3d_case::<f32>();
}

fn avg_pool1d_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&[2, 2, 6], 0.31);
    let w = weighting::<T>(&[2, 2, 4]);
    let layer = AvgPool1d::<T>::with_params(3, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("avg_pool1d backward must match central differences");
}

#[test]
fn avg_pool1d_backward_matches_finite_differences() {
    avg_pool1d_case::<f64>();
    avg_pool1d_case::<f32>();
}

fn avg_pool2d_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&[1, 2, 5, 5], 0.37);
    let w = weighting::<T>(&[1, 2, 2, 2]);
    let layer = AvgPool2d::<T>::with_params(2, 2, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("avg_pool2d backward must match central differences");
}

#[test]
fn avg_pool2d_backward_matches_finite_differences() {
    avg_pool2d_case::<f64>();
    avg_pool2d_case::<f32>();
}

fn avg_pool3d_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&[1, 1, 4, 4, 4], 0.41);
    let w = weighting::<T>(&[1, 1, 2, 2, 2]);
    let layer = AvgPool3d::<T>::with_params(2, 2, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid pooling fixture completes forward");
        weighted(&y, &w)
    })
    .expect("avg_pool3d backward must match central differences");
}

#[test]
fn avg_pool3d_backward_matches_finite_differences() {
    avg_pool3d_case::<f64>();
    avg_pool3d_case::<f32>();
}
