//! Computation graph representation for JIT optimization

use crate::error::{JitError, Result};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Unique identifier for nodes in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Operation types that can be represented in the computation graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    // Arithmetic operations
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,

    // Element-wise operations
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,

    // Matrix operations
    MatMul,
    Transpose,

    // Neural network operations
    Linear,
    Conv2d,
    ReLU,
    Sigmoid,
    Softmax,

    // Tensor operations
    Reshape,
    Slice,
    Concat,

    // Constants and variables
    Constant,
    Parameter,
}

/// Metadata associated with graph nodes
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub requires_grad: bool,
    pub name: Option<String>,
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub metadata: NodeMetadata,
}

impl Node {
    /// Create a new node
    pub fn new(id: NodeId, operation: Operation, metadata: NodeMetadata) -> Self {
        Self {
            id,
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata,
        }
    }

    /// Add an input connection
    pub fn add_input(&mut self, input_id: NodeId) {
        self.inputs.push(input_id);
    }

    /// Add an output connection
    pub fn add_output(&mut self, output_id: NodeId) {
        self.outputs.push(output_id);
    }

    /// Check if this node can be fused with another
    pub fn can_fuse_with(&self, other: &Node) -> bool {
        // Basic fusion rules - can be extended
        match (&self.operation, &other.operation) {
            // Element-wise operations can often be fused
            (Operation::ReLU, Operation::Add) => true,
            (Operation::Add, Operation::ReLU) => true,
            (Operation::Multiply, Operation::Add) => true,
            (Operation::Add, Operation::Multiply) => true,
            // Matrix operations with element-wise
            (Operation::MatMul, Operation::ReLU) => true,
            _ => false,
        }
    }
}

/// Computation graph with optimization capabilities
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    nodes: IndexMap<NodeId, Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    next_id: usize,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, operation: Operation, metadata: NodeMetadata) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let node = Node::new(id, operation, metadata);
        self.nodes.insert(id, node);
        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<()> {
        if !self.nodes.contains_key(&from) {
            return Err(JitError::InvalidGraph {
                message: format!("Source node {:?} does not exist", from),
            });
        }
        if !self.nodes.contains_key(&to) {
            return Err(JitError::InvalidGraph {
                message: format!("Target node {:?} does not exist", to),
            });
        }

        self.nodes[&from].add_output(to);
        self.nodes[&to].add_input(from);
        Ok(())
    }

    /// Mark a node as an input to the graph
    pub fn mark_input(&mut self, node_id: NodeId) {
        if !self.inputs.contains(&node_id) {
            self.inputs.push(node_id);
        }
    }

    /// Mark a node as an output of the graph
    pub fn mark_output(&mut self, node_id: NodeId) {
        if !self.outputs.contains(&node_id) {
            self.outputs.push(node_id);
        }
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get mutable access to a node
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes in topological order
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut order = Vec::new();

        fn visit(
            node_id: NodeId,
            graph: &ComputationGraph,
            visited: &mut HashSet<NodeId>,
            visiting: &mut HashSet<NodeId>,
            order: &mut Vec<NodeId>,
        ) -> Result<()> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            if visiting.contains(&node_id) {
                return Err(JitError::InvalidGraph {
                    message: "Cycle detected in computation graph".to_string(),
                });
            }

            visiting.insert(node_id);

            if let Some(node) = graph.get_node(node_id) {
                for &output_id in &node.outputs {
                    visit(output_id, graph, visited, visiting, order)?;
                }
            }

            visiting.remove(&node_id);
            visited.insert(node_id);
            order.push(node_id);
            Ok(())
        }

        for &input_id in &self.inputs {
            visit(input_id, self, &mut visited, &mut visiting, &mut order)?;
        }

        // Reverse to get topological order (dependencies first)
        order.reverse();
        Ok(order)
    }

    /// Get the number of nodes in the graph
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get iterator over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (&NodeId, &Node)> {
        self.nodes.iter()
    }

    /// Get mutable iterator over all nodes
    pub fn nodes_mut(&mut self) -> impl Iterator<Item = (&NodeId, &mut Node)> {
        self.nodes.iter_mut()
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Get mutable access to input nodes
    pub fn inputs_mut(&mut self) -> &mut Vec<NodeId> {
        &mut self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Get mutable access to output nodes
    pub fn outputs_mut(&mut self) -> &mut Vec<NodeId> {
        &mut self.outputs
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: NodeId) {
        self.nodes.swap_remove(&node_id);
        // Update connections
        for (_, node) in self.nodes.iter_mut() {
            node.inputs.retain(|&id| id != node_id);
            node.outputs.retain(|&id| id != node_id);
        }
        // Remove from inputs/outputs if present
        self.inputs.retain(|&id| id != node_id);
        self.outputs.retain(|&id| id != node_id);
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut graph = ComputationGraph::new();

        let input = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![10]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("input".to_string()),
            },
        );

        let weight = graph.add_node(
            Operation::Parameter,
            NodeMetadata {
                shape: Some(vec![10, 5]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("weight".to_string()),
            },
        );

        let matmul = graph.add_node(
            Operation::MatMul,
            NodeMetadata {
                shape: Some(vec![5]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("matmul".to_string()),
            },
        );

        let relu = graph.add_node(
            Operation::ReLU,
            NodeMetadata {
                shape: Some(vec![5]),
                dtype: Some("f32".to_string()),
                requires_grad: true,
                name: Some("relu".to_string()),
            },
        );

        graph.add_edge(input, matmul).unwrap();
        graph.add_edge(weight, matmul).unwrap();
        graph.add_edge(matmul, relu).unwrap();

        graph.mark_input(input);
        graph.mark_input(weight);
        graph.mark_output(relu);

        assert_eq!(graph.len(), 4);
        assert_eq!(graph.inputs(), &[input, weight]);
        assert_eq!(graph.outputs(), &[relu]);
    }

    #[test]
    fn test_topological_order() {
        let mut graph = ComputationGraph::new();

        let a = graph.add_node(Operation::Parameter, NodeMetadata::default());
        let b = graph.add_node(Operation::Parameter, NodeMetadata::default());
        let c = graph.add_node(Operation::Add, NodeMetadata::default());

        graph.add_edge(a, c).unwrap();
        graph.add_edge(b, c).unwrap();

        graph.mark_input(a);
        graph.mark_input(b);
        graph.mark_output(c);

        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 3);
        // Parameters should come before operations that depend on them
        assert!(
            order.iter().position(|&x| x == a).unwrap()
                < order.iter().position(|&x| x == c).unwrap()
        );
        assert!(
            order.iter().position(|&x| x == b).unwrap()
                < order.iter().position(|&x| x == c).unwrap()
        );
    }

    #[test]
    fn test_fusion_compatibility() {
        let node1 = Node::new(NodeId(1), Operation::MatMul, NodeMetadata::default());
        let node2 = Node::new(NodeId(2), Operation::ReLU, NodeMetadata::default());

        assert!(node1.can_fuse_with(&node2));

        let node3 = Node::new(NodeId(3), Operation::Constant, NodeMetadata::default());
        let node4 = Node::new(NodeId(4), Operation::Parameter, NodeMetadata::default());

        assert!(!node3.can_fuse_with(&node4));
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            shape: None,
            dtype: None,
            requires_grad: false,
            name: None,
        }
    }
}
