//! Generic Coeus pooling dispatch through Hephaestus.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{backward as pooling_backward, forward as pooling_forward};
pub use provider::{PoolingBackend, PoolingProvider};
