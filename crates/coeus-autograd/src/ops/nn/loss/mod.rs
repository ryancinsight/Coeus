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

pub use bce::{BinaryCrossEntropyNode, binary_cross_entropy};
pub use bce_with_logits::{BceWithLogitsNode, bce_with_logits};
pub use cosine::{CosineEmbeddingLossNode, cosine_embedding_loss};
pub use cosine_similarity::{CosineSimilarityNode, cosine_similarity};
pub use cross_entropy::{CrossEntropyLossNode, cross_entropy_loss};
pub use ctc::{CtcLossNode, ctc_loss};
pub use huber::{HuberLossNode, huber_loss};
pub use kl_div::{KlDivLossNode, kl_divergence};
pub use l1::{L1LossNode, l1_loss};
pub use margin_ranking::{MarginRankingLossNode, margin_ranking_loss};
pub use multi_label_margin::{MultiLabelMarginLossNode, multi_label_margin_loss};
pub use multi_margin::{MultiMarginNode, multi_margin};
pub use nll::{NllLossNode, nll_loss};
pub use pairwise_distance::{PairwiseDistanceNode, pairwise_distance};
pub use poisson_nll::{PoissonNllNode, poisson_nll};
pub use smooth_l1::{SmoothL1LossNode, smooth_l1_loss};
pub use soft_margin::{SoftMarginNode, soft_margin};
