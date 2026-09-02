//! Generic Coeus pooling dispatch through Hephaestus.

mod dispatch;
mod implementation;
mod provider;

pub use provider::{PoolingBackend, PoolingProvider};
