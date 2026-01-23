//! Function objects for automatic differentiation hierarchy

pub mod activation;
pub mod arithmetic;
pub mod classification;
pub mod layout;
pub mod linalg;
pub mod math;
pub mod reduction;
pub mod rnn;

pub use activation::*;
pub use arithmetic::*;
pub use classification::*;
pub use layout::*;
pub use linalg::*;
pub use math::*;
pub use reduction::*;
pub use rnn::*;

// Helper functions for backward passes (moved from autograd or legacy functions.rs)
pub mod utils;
pub use utils::*;
