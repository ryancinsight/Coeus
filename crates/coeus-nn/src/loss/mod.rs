//! Loss functions for classification, regression, and sequence alignment.
//!
//! One module per loss family: regression (elementwise prediction error),
//! classification (class-label and distribution losses), margin (hinge and
//! ranking objectives), metric (distance and similarity losses), nan
//! (NaN-aware reductions), and ctc (sequence alignment). Every entry point
//! is re-exported here, so `coeus_nn::loss::*` and the crate-root re-exports
//! are the stable public surface.

mod classification;
mod ctc;
mod margin;
mod metric;
mod nan;
mod regression;

pub use classification::{
    bce_with_logits, binary_cross_entropy, cross_entropy_loss, kl_divergence,
    multi_label_soft_margin_loss, nll_loss,
};
pub use ctc::ctc_loss;
pub use margin::{
    hinge_embedding_loss, margin_ranking_loss, multi_label_margin_loss, multi_margin, soft_margin,
};
pub use metric::{
    cosine_embedding_loss, cosine_similarity, pairwise_distance, triplet_margin_loss,
    triplet_margin_with_distance_loss,
};
pub use nan::{nanmean, nansum};
pub use regression::{
    gaussian_nll_loss, huber_loss, l1_loss, mse_loss, poisson_nll, smooth_l1_loss,
};
