/// Stateless activation functions (ReLU, GELU, SiLU, etc.).
pub mod basic;
/// Parametric activation functions (ELU, PReLU, etc.).
pub mod parametric;

pub use basic::*;
pub use parametric::*;
