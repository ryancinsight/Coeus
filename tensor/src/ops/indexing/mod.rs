//! Indexing operations for tensors
//!
//! This module provides advanced indexing operations that manipulate
//! tensor elements based on index tensors.
//!
//! ## Operations
//!
//! - `gather` - Gather elements along a dimension using indices
//! - `scatter` - Scatter values into tensor at given indices
//! - `index_select` - Select elements from a dimension
//! - `masked_fill` - Fill elements where mask is true

pub mod gather;
pub mod scatter;
pub mod index_select;

pub use gather::gather;
pub use scatter::scatter;
pub use index_select::index_select;
