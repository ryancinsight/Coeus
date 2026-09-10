//! Caching integration for backward pass compilation.
//!
//! The metadata and topology implementations live in focused leaf modules so
//! each cache operation has one canonical home. The cache retains a bounded
//! topology plan for a live graph while sharing structural metadata across
//! separately constructed graphs.

mod fingerprint;
mod topological_sort;

pub use fingerprint::compute_graph_structure_fingerprint;
pub use topological_sort::topological_sort_with_cache;

// The fixture modules exercise the single traversal directly to verify that
// metadata-only and topology-producing calls share the same implementation.
#[cfg(test)]
use crate::autodiff_cache::{ComputeGraphCache, GraphInfo};
#[cfg(test)]
use crate::node::BackwardNode;
#[cfg(test)]
pub(super) use fingerprint::collect_graph;
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
mod tests;
