//! Generic Coeus-to-Hephaestus attention dispatch.

mod dispatch;
mod implementation;
mod provider;

pub use provider::{AttentionBackend, AttentionProvider};
