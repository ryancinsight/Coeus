//! Distance- and similarity-based losses.
//!
//! Metric losses over embedding rows: pairwise p-norm distance, cosine
//! similarity and its embedding loss, and the triplet-margin family built on
//! pluggable distance functions.

use coeus_autograd::Var;
use coeus_core::Float;

/// Row-wise p-norm pairwise distance (PyTorch `PairwiseDistance`).
/// x1, x2: `[N, D]`; returns `[N]` with `out_i = (sum_k |x1-x2|^p + eps)^(1/p)`.
#[inline]
pub fn pairwise_distance<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    p: T,
    eps: T,
) -> Var<T, B> {
    coeus_autograd::pairwise_distance(x1, x2, p, eps)
}

/// Triplet-margin loss (PyTorch `TripletMarginLoss`, `reduction="mean"`):
/// `mean_i max(0, d(a_i, p_i) - d(a_i, n_i) + margin)` where `d` is the
/// row-wise p-norm [`pairwise_distance`]. anchor/positive/negative share shape
/// `[N, D]`. Composed from the tracked pairwise-distance, subtract, shift, ReLU,
/// and mean ops, so backward (including the anchor's two gradient paths) is the
/// autograd graph's — no bespoke node.
#[inline]
pub fn triplet_margin_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    anchor: &Var<T, B>,
    positive: &Var<T, B>,
    negative: &Var<T, B>,
    margin: T,
    p: T,
    eps: T,
) -> Var<T, B> {
    let d_ap = coeus_autograd::pairwise_distance(anchor, positive, p, eps);
    let d_an = coeus_autograd::pairwise_distance(anchor, negative, p, eps);
    let shifted = coeus_autograd::scalar_add(&coeus_autograd::sub(&d_ap, &d_an), margin);
    coeus_autograd::mean(&coeus_autograd::relu(&shifted))
}

/// Triplet margin loss with pluggable distance function
/// (PyTorch `TripletMarginWithDistanceLoss`, `reduction="mean"`).
///
/// Generalizes `triplet_margin_loss` by accepting a custom distance function.
/// `distance(a, p)` and `distance(a, n)` are computed via the provided closure.
/// Returns `mean(max(0, d_ap - d_an + margin))`.
pub fn triplet_margin_with_distance_loss<T, B, F>(
    anchor: &Var<T, B>,
    positive: &Var<T, B>,
    negative: &Var<T, B>,
    distance: F,
    margin: T,
) -> Var<T, B>
where
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
    F: Fn(&Var<T, B>, &Var<T, B>) -> Var<T, B>,
{
    let d_ap = distance(anchor, positive);
    let d_an = distance(anchor, negative);
    let shifted = coeus_autograd::scalar_add(&coeus_autograd::sub(&d_ap, &d_an), margin);
    coeus_autograd::mean(&coeus_autograd::relu(&shifted))
}

/// Cosine Embedding Loss.
/// x1: `[N, D]`, x2: `[N, D]`, y: `[N]`, margin: threshold.
#[inline]
pub fn cosine_embedding_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    y: &[T],
    margin: T,
) -> Var<T, B> {
    coeus_autograd::cosine_embedding_loss(x1, x2, y, margin)
}

/// Row-wise cosine similarity along `dim=1`
/// (PyTorch `F.cosine_similarity(x1, x2, dim=1, eps=...)`).
/// `x1` and `x2` must share shape `[N, D]`; returns `[N]` where
/// `out_i = <x1_i, x2_i> / max(||x1_i|| * ||x2_i||, eps)`.
///
/// # Panics
///
/// Panics when the inputs do not share a two-dimensional non-empty shape,
/// `dim` is not one, or `eps` is not finite and strictly positive.
#[must_use]
#[inline]
pub fn cosine_similarity<
    T: coeus_core::FloatElement + Float,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    dim: usize,
    eps: T,
) -> Var<T, B> {
    coeus_autograd::cosine_similarity(x1, x2, dim, eps)
}
