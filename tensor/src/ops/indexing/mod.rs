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
//! - `take` - Select values from tensor using indices
//! - `put` - Place values into tensor at indices

pub mod gather;
pub mod index_select;
pub mod scatter;
pub mod masked;
pub mod take;
pub mod put;

pub use gather::gather;
pub use index_select::index_select;
pub use scatter::scatter;
pub use take::take;
pub use put::put;
