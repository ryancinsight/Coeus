//! Generic Coeus-to-Hephaestus dense product dispatch.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::matmul;
pub use provider::{MatmulBackend, MatmulProvider};
