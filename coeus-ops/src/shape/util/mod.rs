//! Shape utility helpers shared by tensor construction and indexing code.
//!
//! Includes PyTorch/NumPy-style helpers such as coordinate generation, sorting,
//! and index extraction utilities.

mod meshgrid;
mod nonzero;
mod sort;

pub use meshgrid::meshgrid;
pub use nonzero::nonzero;
pub use sort::sort;
