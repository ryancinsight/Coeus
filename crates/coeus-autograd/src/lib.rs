//! Reverse-mode automatic differentiation engine built on the Coeus tensor and ops stacks.
//!
//! # Key types
//! - [`Var<T, B>`](var::Var) — a tracked tensor carrying an optional gradient accumulator and
//!   an optional `Arc<dyn BackwardNode<T, B>>` creator link.
//! - [`BackwardNode`] — trait implemented by per-op nodes; each node stores
//!   saved tensors and accumulates gradients into its inputs during the reverse pass.
//! - [`Var::backward`](var::Var::backward) — triggers topological traversal of the computation
//!   graph seeded with a ones tensor, propagating gradients to all `requires_grad` leaves.
//!
//! All differentiable ops in [`ops`] are thin wrappers that call [`ops::arithmetic::binary_op`] or
//! [`ops::activation::unary_op`], which construct the forward result and attach the creator node.

// ── Coeus Autograd ──
// Automatic differentiation engine with computational graph.
#![deny(missing_docs)]

/// Computation graph caching for autodiff compilation overhead reduction.
pub mod autodiff_cache;
/// Backward-pass graph traversal and caching integration.
pub mod backward;
pub mod backward_cache;
pub(crate) mod grad_buffer;
/// Thread-local autograd recording mode (no-grad scopes).
pub mod grad_mode;
/// Backward-pass verification against central finite differences.
pub mod gradcheck;
/// Computation graph node trait and implementations.
pub mod node;
/// Differentiable operations that build the autograd graph.
pub mod ops;
/// Named trainable parameter carrier.
pub mod parameter;
/// The differentiable variable type.
pub mod var;

pub use autodiff_cache::{
    compute_graph_fingerprint, CacheConfig, CacheSnapshot, CacheStats, ComputeGraphCache,
    ComputeGraphKey, DefaultCacheConfig, GraphInfo, MemoryBreakdown, TopologyPlanSnapshot,
    PLAN_PURGE_MIN_TABLE_SIZE,
};
pub use backward_cache::topological_sort_with_cache;
pub use grad_buffer::GradBuffer;
pub use grad_mode::{
    is_grad_enabled, is_no_grad_enabled, no_grad_guard, pop_no_grad, push_no_grad, NoGradGuard,
};
pub use gradcheck::{gradcheck, gradcheck_with, GradcheckConfig, GradcheckError};
pub use node::BackwardNode;
/// Differentiable operations. `ops` curates its own public surface, so the
/// crate root re-exports it wholesale rather than repeating the 188-name list.
pub use ops::*;
pub use parameter::Parameter;
pub use var::{get_backward_cache, reset_backward_cache_stats};

pub use var::Var;
