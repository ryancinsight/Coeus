/// Binary cross-entropy loss.
pub mod bce;
/// Binary cross-entropy from logits (numerically stable).
pub mod bce_with_logits;
/// Cosine embedding loss.
pub mod cosine;
/// Cross-entropy loss.
pub mod cross_entropy;
/// Huber loss.
pub mod huber;
/// KL divergence loss.
pub mod kl_div;
/// L1 (mean absolute error) loss.
pub mod l1;
/// Margin ranking loss.
pub mod margin_ranking;
/// Negative log-likelihood loss.
pub mod nll;
/// Poisson negative-log-likelihood loss.
pub mod poisson_nll;
/// Soft-margin (logistic) loss.
pub mod soft_margin;

pub use bce::{binary_cross_entropy, BinaryCrossEntropyNode};
pub use bce_with_logits::{bce_with_logits, BceWithLogitsNode};
pub use cosine::{cosine_embedding_loss, CosineEmbeddingLossNode};
pub use cross_entropy::{cross_entropy_loss, CrossEntropyLossNode};
pub use huber::{huber_loss, HuberLossNode};
pub use kl_div::{kl_divergence, KlDivLossNode};
pub use l1::{l1_loss, L1LossNode};
pub use margin_ranking::{margin_ranking_loss, MarginRankingLossNode};
pub use nll::{nll_loss, NllLossNode};
pub use poisson_nll::{poisson_nll, PoissonNllNode};
pub use soft_margin::{soft_margin, SoftMarginNode};
