//! Dropout layers for regularization.

pub mod spatial;
pub mod standard;

#[cfg(test)]
pub mod tests;

pub use spatial::{Dropout2d, Dropout3d};
pub use standard::Dropout;
