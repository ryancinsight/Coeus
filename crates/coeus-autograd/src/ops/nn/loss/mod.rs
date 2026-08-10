/// Binary cross-entropy loss.
pub mod bce;
/// Binary cross-entropy from logits (numerically stable).
pub mod bce_with_logits;
/// Cosine embedding loss.
pub mod cosine;
/// Row-wise cosine similarity (PyTorch `F.cosine_similarity`).
pub mod cosine_similarity;
/// Cross-entropy loss.
pub mod cross_entropy;
/// CTC (Connectionist Temporal Classification) loss.
pub mod ctc;
/// Huber loss.
pub mod huber;
/// KL divergence loss.
pub mod kl_div;
/// L1 (mean absolute error) loss.
pub mod l1;
/// Global and per-axis Lp-norm (tracked `l2_norm` / `l_p_norm` / `l_p_norm_axis`).
pub mod lp_norm;
/// Margin ranking loss.
pub mod margin_ranking;
/// Multi-label margin loss.
pub mod multi_label_margin;
/// Multi-class margin loss.
pub mod multi_margin;
/// Negative log-likelihood loss.
pub mod nll;
/// Row-wise p-norm pairwise distance.
pub mod pairwise_distance;
/// Poisson negative-log-likelihood loss.
pub mod poisson_nll;
/// Smooth L1 (Huber-β) loss.
pub mod smooth_l1;
/// Soft-margin (logistic) loss.
pub mod soft_margin;

pub use bce::{binary_cross_entropy, BinaryCrossEntropyNode};
pub use bce_with_logits::{bce_with_logits, BceWithLogitsNode};
pub use cosine::{cosine_embedding_loss, CosineEmbeddingLossNode};
pub use cosine_similarity::{cosine_similarity, CosineSimilarityNode};
pub use cross_entropy::{cross_entropy_loss, CrossEntropyLossNode};
pub use ctc::{ctc_loss, CtcLossNode};
pub use huber::{huber_loss, HuberLossNode};
pub use kl_div::{kl_divergence, KlDivLossNode};
pub use l1::{l1_loss, L1LossNode};
pub use lp_norm::{l2_norm, l_p_norm, l_p_norm_axis, LpNormAxisNode, LpNormNode};
pub use margin_ranking::{margin_ranking_loss, MarginRankingLossNode};
pub use multi_label_margin::{multi_label_margin_loss, MultiLabelMarginLossNode};
pub use multi_margin::{multi_margin, MultiMarginNode};
pub use nll::{nll_loss, NllLossNode};
pub use pairwise_distance::{pairwise_distance, PairwiseDistanceNode};
pub use poisson_nll::{poisson_nll, PoissonNllNode};
pub use smooth_l1::{smooth_l1_loss, SmoothL1LossNode};
pub use soft_margin::{soft_margin, SoftMarginNode};
