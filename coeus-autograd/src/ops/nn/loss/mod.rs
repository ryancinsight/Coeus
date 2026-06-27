/// Binary cross-entropy loss.
pub mod bce;
/// Cosine embedding loss.
pub mod cosine;
/// Cross-entropy loss.
pub mod cross_entropy;
/// Huber loss.
pub mod huber;
/// KL divergence loss.
pub mod kl_div;
/// Margin ranking loss.
pub mod margin_ranking;
/// Negative log-likelihood loss.
pub mod nll;

pub use bce::{binary_cross_entropy, BinaryCrossEntropyNode};
pub use cosine::{cosine_embedding_loss, CosineEmbeddingLossNode};
pub use cross_entropy::{cross_entropy_loss, CrossEntropyLossNode};
pub use huber::{huber_loss, HuberLossNode};
pub use kl_div::{kl_divergence, KlDivLossNode};
pub use margin_ranking::{margin_ranking_loss, MarginRankingLossNode};
pub use nll::{nll_loss, NllLossNode};
