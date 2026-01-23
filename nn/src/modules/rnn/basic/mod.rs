//! Basic RNN (Recurrent Neural Network) module.

pub mod core;
pub mod display;
pub mod forward;
pub mod module;

#[cfg(test)]
pub mod tests;

pub use core::RNN;
