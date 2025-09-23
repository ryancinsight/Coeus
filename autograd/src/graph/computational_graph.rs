//! Computational graph implementation for automatic differentiation
//!
//! This module contains the ComputationalGraph struct and core graph operations.

use crate::{AutogradError, Node, NodeId, TensorRef};
use crate::context::Operation;
use coeus_dtype::Dtype;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Computational graph for automatic differentiation
pub struct ComputationalGraph<T: Dtype> {
    /// All nodes in the graph
    nodes: HashMap<NodeId, Node<T>>,
    /// Adjacency list for forward dependencies
    forward_deps: HashMap<NodeId, Vec<NodeId>>,
    /// Adjacency list for backward dependencies
    backward_deps: HashMap<NodeId, Vec<NodeId>>,
    /// Next available node ID
    next_id: usize,
    /// Optimization: cache for topological sort
    topo_cache: Option<Vec<NodeId>>,
    /// Optimization: cache for gradient computations
    grad_cache: HashMap<NodeId, TensorRef<T>>,
}

impl<T> Clone for ComputationalGraph<T>
where
    T: Dtype + std::ops::Neg<Output = T>,
{
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            forward_deps: self.forward_deps.clone(),
            backward_deps: self.backward_deps.clone(),
            next_id: self.next_id,
            topo_cache: self.topo_cache.clone(),
            grad_cache: self.grad_cache.clone(),
        }
    }
}

impl<T> ComputationalGraph<T>
where
    T: Dtype + std::ops::Neg<Output = T>,
{
    /// Create a new computational graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            forward_deps: HashMap::new(),
            backward_deps: HashMap::new(),
            next_id: 0,
            topo_cache: None,
            grad_cache: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node<T>) {
        let node_id = node.id;

        // Update dependencies
        // TODO: Implement dependency tracking for ComputationalGraph
        // The Operation enum doesn't contain inputs, so this needs to be redesigned
        // or the operation inputs need to be tracked separately

        self.nodes.insert(node_id, node);
        self.invalidate_cache();
    }

    /// Create a new node with automatic ID assignment
    pub fn create_node(
        &mut self,
        data: TensorRef<T>,
        operation: Option<Operation>,
        requires_grad: bool,
    ) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let node = Node::new(id, data, operation, requires_grad);
        self.add_node(node);
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: &NodeId) -> Option<&Node<T>> {
        self.nodes.get(id)
    }

    /// Get a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, id: &NodeId) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, Node<T>> {
        &self.nodes
    }

    /// Check if node exists
    pub fn contains_node(&self, id: &NodeId) -> bool {
        self.nodes.contains_key(id)
    }

    /// Perform topological sort for backward pass
    pub fn topological_sort(&self, start_nodes: &[NodeId]) -> Result<Vec<NodeId>, AutogradError> {
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut order = Vec::new();

        for &node_id in start_nodes {
            if !visited.contains(&node_id) {
                self.topological_sort_helper(node_id, &mut visited, &mut visiting, &mut order)?;
            }
        }

        Ok(order)
    }

    fn topological_sort_helper(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        visiting: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<(), AutogradError> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        if visiting.contains(&node_id) {
            return Err(AutogradError::CycleDetected);
        }

        visiting.insert(node_id);

        // Visit all dependencies
        if let Some(deps) = self.backward_deps.get(&node_id) {
            for &dep_id in deps {
                self.topological_sort_helper(dep_id, visited, visiting, order)?;
            }
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        order.push(node_id);

        Ok(())
    }

    /// Compute gradients using reverse-mode automatic differentiation
    pub fn backward(&mut self, output_nodes: &[NodeId]) -> Result<(), AutogradError> {
        // Reset all gradients
        for node in self.nodes.values_mut() {
            node.grad = None;
        }

        // Initialize output gradients to 1.0
        for &output_id in output_nodes {
            if let Some(node) = self.nodes.get_mut(&output_id) {
                let ones = TensorRef::ones(node.data.shape().to_vec());
                node.set_grad(ones);
            }
        }

        // Get topological order
        let topo_order = self.topological_sort(output_nodes)?;
        let mut processed = HashSet::new();

        // Process nodes in reverse topological order
        for &node_id in topo_order.iter().rev() {
            if processed.contains(&node_id) {
                continue;
            }

            // TODO: Implement backward propagation for ComputationalGraph
            // This code was broken during modularization and needs to be rewritten
            // to work with the new Operation enum instead of the Operation struct
            if let Some(_node) = self.nodes.get(&node_id) {
                // Backward propagation not implemented for ComputationalGraph
                // The active autograd system uses AutogradContext
            }

            processed.insert(node_id);
        }

        Ok(())
    }

    /// Clear gradient cache
    pub fn clear_grad_cache(&mut self) {
        self.grad_cache.clear();
        for node in self.nodes.values_mut() {
            node.grad = None;
        }
    }

    /// Invalidate caches when graph structure changes
    fn invalidate_cache(&mut self) {
        self.topo_cache = None;
        self.grad_cache.clear();
    }

    /// Get gradient for a node
    pub fn get_gradient(&self, node_id: &NodeId) -> Option<&TensorRef<T>> {
        self.nodes.get(node_id).and_then(|node| node.grad.as_ref())
    }

    /// Remove a node and its dependencies
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        if self.nodes.remove(node_id).is_some() {
            // Remove from dependency lists
            self.forward_deps.remove(node_id);
            self.backward_deps.remove(node_id);

            // Remove this node from other nodes' dependency lists
            for deps in self.forward_deps.values_mut() {
                deps.retain(|&id| id != *node_id);
            }
            for deps in self.backward_deps.values_mut() {
                deps.retain(|&id| id != *node_id);
            }

            self.invalidate_cache();
            true
        } else {
            false
        }
    }

    /// Get the number of nodes in the graph
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T> Default for ComputationalGraph<T>
where
    T: Dtype + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Dtype> fmt::Debug for ComputationalGraph<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputationalGraph")
            .field("nodes", &self.nodes.len())
            .field("forward_deps", &self.forward_deps.len())
            .field("backward_deps", &self.backward_deps.len())
            .finish()
    }
}
