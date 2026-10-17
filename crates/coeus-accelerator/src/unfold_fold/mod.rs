//! Generic Coeus unfold/fold dispatch through Hephaestus.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{fold as unfold_fold_fold, unfold as unfold_fold_unfold};
pub use provider::{UnfoldFoldBackend, UnfoldFoldProvider};
