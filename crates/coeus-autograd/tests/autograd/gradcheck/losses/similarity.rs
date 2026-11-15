//! Finite-difference checks for the distance and embedding losses.

use super::super::{weighted, weighting};
use super::{GradcheckScalar, Sampler};
use coeus_autograd::{cosine_embedding_loss, cosine_similarity, gradcheck, pairwise_distance};

fn cosine_similarity_case<T: GradcheckScalar>() {
    // <a,b>/(||a||·||b||) is smooth away from zero rows; both operands are
    // sampled strictly positive-magnitude, so neither norm approaches the eps
    // clamp and the full norm-derivative path — not the clamped constant one —
    // is the branch under test.
    let x1 = Sampler::new(0.13, 0.3, 1.6).tensor::<T>(&[3, 4]);
    let x2 = Sampler::new(0.79, -1.6, -0.3).tensor::<T>(&[3, 4]);
    let w = weighting::<T>(&[3]);

    gradcheck(&[x1, x2], |v| {
        weighted(
            &cosine_similarity(
                &v[0],
                &v[1],
                1,
                <T as coeus_core::FloatElement>::from_f64(1e-8),
            )
            .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("cosine_similarity backward must match central differences");
}

#[test]
fn cosine_similarity_backward_matches_finite_differences() {
    cosine_similarity_case::<f64>();
    cosine_similarity_case::<f32>();
}

fn cosine_embedding_loss_case<T: GradcheckScalar>() {
    // y = +1 rows contribute 1 - cos, y = -1 rows contribute max(0, cos -
    // margin). With margin = -0.5 and the operands below anti-aligned, the
    // y = -1 hinge is firmly active, so both the smooth and the hinge branch are
    // covered without either sitting on its break.
    let x1 = Sampler::new(0.13, 0.3, 1.6).tensor::<T>(&[3, 4]);
    let x2 = Sampler::new(0.79, -1.6, -0.3).tensor::<T>(&[3, 4]);
    let labels: Vec<T> = [1.0, -1.0, 1.0]
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();

    gradcheck(&[x1, x2], |v| {
        cosine_embedding_loss(
            &v[0],
            &v[1],
            &labels,
            <T as coeus_core::FloatElement>::from_f64(-0.5),
        )
        .expect("invariant: test operation succeeds")
    })
    .expect("cosine_embedding_loss backward must match central differences");
}

#[test]
fn cosine_embedding_loss_backward_matches_finite_differences() {
    cosine_embedding_loss_case::<f64>();
    cosine_embedding_loss_case::<f32>();
}

fn pairwise_distance_case<T: GradcheckScalar>() {
    // p = 2 keeps |x1 - x2 + eps|^p smooth through zero, so this checks the
    // s^(1/p - 1) chain factor without a kink in the way. The operands are drawn
    // from disjoint intervals, so every difference is bounded away from zero and
    // the row sums stay well clear of the origin where the p-norm is
    // non-differentiable.
    let x1 = Sampler::new(0.23, 0.4, 1.7).tensor::<T>(&[3, 4]);
    let x2 = Sampler::new(0.61, -1.7, -0.4).tensor::<T>(&[3, 4]);
    let w = weighting::<T>(&[3]);

    gradcheck(&[x1, x2], |v| {
        weighted(
            &pairwise_distance(
                &v[0],
                &v[1],
                <T as coeus_core::FloatElement>::from_f64(2.0),
                <T as coeus_core::FloatElement>::from_f64(1e-6),
            )
            .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("pairwise_distance p=2 backward must match central differences");
}

#[test]
fn pairwise_distance_backward_matches_finite_differences() {
    pairwise_distance_case::<f64>();
    pairwise_distance_case::<f32>();
}

fn pairwise_distance_p_three_case<T: GradcheckScalar>() {
    // p = 3 exercises the general |d|^(p-1)·sign(d) path rather than the p = 2
    // special case, where the sign factor cancels and an implementation can be
    // wrong without the p = 2 check noticing.
    let x1 = Sampler::new(0.37, 0.4, 1.7).tensor::<T>(&[2, 3]);
    let x2 = Sampler::new(0.83, -1.7, -0.4).tensor::<T>(&[2, 3]);
    let w = weighting::<T>(&[2]);

    gradcheck(&[x1, x2], |v| {
        weighted(
            &pairwise_distance(
                &v[0],
                &v[1],
                <T as coeus_core::FloatElement>::from_f64(3.0),
                <T as coeus_core::FloatElement>::from_f64(1e-6),
            )
            .expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("pairwise_distance p=3 backward must match central differences");
}

#[test]
fn pairwise_distance_backward_matches_finite_differences_at_p_three() {
    pairwise_distance_p_three_case::<f64>();
    pairwise_distance_p_three_case::<f32>();
}
