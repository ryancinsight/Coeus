//! Computation graph caching for autodiff.
//!
//! This module provides a memoization cache for computation graphs to reduce
//! autodiff compilation overhead in iterative solvers by caching and reusing
//! computation graph topologies for repeated operation patterns.
//!
//! # Design
//!
//! The cache works by:
//! 1. Computing a fingerprint of the computation graph structure
//! 2. Keying on input shapes, operation sequence, and backend type
//! 3. Storing graph metadata for repeated-pattern accounting
//! 4. Using LRU eviction with generation-based validation
//!
//! # Performance
//!
//! - Cache hits: reuse graph metadata and, for the same live root, its topology plan
//! - Cache misses: compute graph metadata and cache it for future iterations
//! - Topology plans retain weak node references, so dropped graphs are not kept alive
//! - Structurally equivalent but separately allocated graphs still build separate plans

mod cache;
mod config;
mod fingerprint;
mod key;
mod plan;
mod plans;
mod stats;

pub use cache::ComputeGraphCache;
pub use config::{CacheConfig, DefaultCacheConfig};
pub use fingerprint::compute_graph_fingerprint;
pub use key::{ComputeGraphKey, GraphInfo};
pub use stats::{
    CacheSnapshot, CacheStats, MemoryBreakdown, TopologyPlanSnapshot, PLAN_PURGE_MIN_TABLE_SIZE,
};
