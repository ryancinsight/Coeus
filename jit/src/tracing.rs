//! Tracing integration for JIT compilation
//!
//! This module provides hooks into the autograd system to record operations
//! during forward passes for JIT compilation. It enables TorchScript-style
//! tracing where model execution is recorded into computation graphs.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::{JitError, Result};
use crate::graph::{ComputationGraph, NodeId, NodeMetadata, Operation};

thread_local! {
    static TRACING_CONTEXT: RefCell<Option<TracingContext>> = const { RefCell::new(None) };
}

/// Context for tracing autograd operations into JIT graphs
#[derive(Debug)]
#[allow(dead_code)]
pub struct TracingContext {
    /// The computation graph being built
    graph: ComputationGraph,
    /// Mapping from autograd variable IDs to JIT node IDs
    variable_map: HashMap<usize, NodeId>,
    /// Next available node ID
    next_node_id: NodeId,
}

impl TracingContext {
    /// Create a new tracing context
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            variable_map: HashMap::new(),
            next_node_id: NodeId(0),
        }
    }

    /// Record an operation in the computation graph
    ///
    /// # Arguments
    /// * `operation` - The operation to record
    /// * `input_vars` - IDs of input variables
    /// * `output_var` - ID of the output variable
    /// * `metadata` - Node metadata for the operation
    ///
    /// # Returns
    /// The node ID of the recorded operation
    pub fn record_operation(
        &mut self,
        operation: Operation,
        input_vars: Vec<usize>,
        output_var: usize,
        metadata: NodeMetadata,
    ) -> NodeId {
        // Map autograd variable IDs to JIT node IDs
        let input_nodes: Vec<NodeId> = input_vars
            .iter()
            .filter_map(|&var_id| self.variable_map.get(&var_id).copied())
            .collect();

        // Create the operation node
        let op_node_id = self.graph.add_node(operation, metadata);

        // Create output variable node
        let output_node_id = self.graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: None, // Will be inferred
                dtype: None,
                requires_grad: true,
                name: Some(format!("var_{}", output_var)),
            },
        );

        // Add edges from inputs to operation
        for &input_node in &input_nodes {
            self.graph.add_edge(input_node, op_node_id).unwrap();
        }

        // Add edge from operation to output
        self.graph.add_edge(op_node_id, output_node_id).unwrap();

        // Mark the output as a graph output
        self.graph.mark_output(output_node_id);

        // Record the mapping
        self.variable_map.insert(output_var, output_node_id);

        op_node_id
    }

    /// Record a parameter/input variable
    ///
    /// # Arguments
    /// * `var_id` - Autograd variable ID
    /// * `metadata` - Node metadata
    ///
    /// # Returns
    /// The JIT node ID for this variable
    pub fn record_parameter(&mut self, var_id: usize, metadata: NodeMetadata) -> NodeId {
        let node_id = self.graph.add_node(Operation::Parameter, metadata);
        self.graph.mark_input(node_id);

        self.variable_map.insert(var_id, node_id);
        node_id
    }

    /// Get the recorded computation graph
    pub fn into_graph(self) -> ComputationGraph {
        self.graph
    }
}

/// RAII guard for tracing context
pub struct TracingGuard {
    _private: (),
}

impl TracingGuard {
    /// Start tracing operations
    pub fn start_tracing() -> Result<Self> {
        TRACING_CONTEXT.with(|ctx| {
            if ctx.borrow().is_some() {
                return Err(JitError::TracingError {
                    message: "Tracing already active".to_string(),
                });
            }

            *ctx.borrow_mut() = Some(TracingContext::new());
            Ok(Self { _private: () })
        })
    }

    /// Stop tracing and return the recorded graph
    pub fn stop_tracing(self) -> Result<ComputationGraph> {
        TRACING_CONTEXT.with(|ctx| {
            if let Some(tracing_ctx) = ctx.borrow_mut().take() {
                Ok(tracing_ctx.into_graph())
            } else {
                Err(JitError::TracingError {
                    message: "No active tracing context".to_string(),
                })
            }
        })
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        // Clean up tracing context when guard goes out of scope
        TRACING_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });
    }
}

/// Check if tracing is currently active
pub fn is_tracing() -> bool {
    TRACING_CONTEXT.with(|ctx| ctx.borrow().is_some())
}

/// Record an operation if tracing is active
///
/// # Arguments
/// * `operation` - The operation to record
/// * `input_vars` - IDs of input variables
/// * `output_var` - ID of the output variable
/// * `metadata` - Node metadata
///
/// # Returns
/// Some(node_id) if tracing was active, None otherwise
pub fn record_operation(
    operation: Operation,
    input_vars: Vec<usize>,
    output_var: usize,
    metadata: NodeMetadata,
) -> Option<NodeId> {
    TRACING_CONTEXT.with(|ctx| {
        (*ctx.borrow_mut()).as_mut().map(|tracing_ctx| {
            tracing_ctx.record_operation(operation, input_vars, output_var, metadata)
        })
    })
}

/// Record a parameter if tracing is active
///
/// # Arguments
/// * `var_id` - Autograd variable ID
/// * `metadata` - Node metadata
///
/// # Returns
/// Some(node_id) if tracing was active, None otherwise
pub fn record_parameter(var_id: usize, metadata: NodeMetadata) -> Option<NodeId> {
    TRACING_CONTEXT.with(|ctx| {
        (*ctx.borrow_mut())
            .as_mut()
            .map(|tracing_ctx| tracing_ctx.record_parameter(var_id, metadata))
    })
}

/// Get the current tracing context (for internal use)
#[allow(dead_code)]
pub(crate) fn with_tracing_context<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut TracingContext) -> R,
{
    TRACING_CONTEXT.with(|ctx| (*ctx.borrow_mut()).as_mut().map(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_context_creation() {
        let ctx = TracingContext::new();
        assert_eq!(ctx.next_node_id.0, 0);
        assert!(ctx.variable_map.is_empty());
    }

    #[test]
    fn test_tracing_guard() {
        assert!(!is_tracing());

        {
            let _guard = TracingGuard::start_tracing().unwrap();
            assert!(is_tracing());
        }

        assert!(!is_tracing());
    }

    #[test]
    fn test_nested_tracing_guard_fails() {
        let _guard1 = TracingGuard::start_tracing().unwrap();
        assert!(TracingGuard::start_tracing().is_err());
    }

    #[test]
    fn test_record_parameter() {
        let mut ctx = TracingContext::new();

        let node_id = ctx.record_parameter(42, NodeMetadata::default());
        assert_eq!(node_id.0, 0);
        assert_eq!(ctx.variable_map[&42], node_id);

        // Graph should have one input node
        assert_eq!(ctx.graph.inputs().len(), 1);
        assert_eq!(ctx.graph.outputs().len(), 0);
    }

    #[test]
    fn test_record_operation() {
        let mut ctx = TracingContext::new();

        // Record input parameters first
        let input1 = ctx.record_parameter(1, NodeMetadata::default());
        let input2 = ctx.record_parameter(2, NodeMetadata::default());

        // Record an operation
        let op_node = ctx.record_operation(
            Operation::Add,
            vec![1, 2],
            3,
            NodeMetadata {
                shape: Some(vec![4]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("add".to_string()),
            },
        );

        assert_eq!(op_node.0, 2); // After two parameter nodes

        // Check that the graph has proper structure
        assert_eq!(ctx.graph.inputs().len(), 2);
        assert_eq!(ctx.graph.outputs().len(), 1);

        // Check variable mapping
        assert_eq!(ctx.variable_map[&3].0, 3); // Output variable node
    }

    #[test]
    fn test_tracing_integration() {
        // Start tracing
        assert!(!is_tracing());
        let guard = TracingGuard::start_tracing().unwrap();
        assert!(is_tracing());

        // Record operations
        let param_id = record_parameter(1, NodeMetadata::default()).unwrap();
        let op_id = record_operation(
            Operation::ReLU,
            vec![1],
            2,
            NodeMetadata {
                shape: Some(vec![10]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("relu".to_string()),
            },
        )
        .unwrap();

        assert_eq!(param_id.0, 0);
        assert_eq!(op_id.0, 1);

        // Stop tracing and get graph
        let graph = guard.stop_tracing().unwrap();
        assert!(!is_tracing());

        // Verify graph structure
        assert_eq!(graph.inputs().len(), 1);
        assert_eq!(graph.outputs().len(), 1);
        assert_eq!(graph.nodes().count(), 3); // input param + operation + output param
    }
}
