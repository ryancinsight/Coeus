//! Modular computational graph implementation
//!
//! This module provides a modular implementation of the computational graph
//! for automatic differentiation, split into focused submodules.

pub mod computational_graph;
pub mod hessian;
pub mod node;
pub mod operation;

// Re-export main types for convenience
pub use computational_graph::ComputationalGraph;
pub use hessian::HessianIter;
pub use node::{Node, NodeId};
pub use operation::Operation;

#[cfg(test)]
mod tests;
