//! Node-based computation graph for automatic differentiation
//!
//! This module implements a node-based computation graph architecture that
//! enables dynamic graph construction compatible with `PyTorch`'s autograd system.

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::DenseStorage;
use coeus_tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Unique identifier for nodes in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    /// Create a new unique node ID
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

/// Operation type for differentiable computations
#[derive(Debug, Clone)]
pub enum Operation {
    /// Addition: a + b
    Add,
    /// Subtraction: a - b
    Sub,
    /// Multiplication: a * b
    Mul,
    /// Division: a / b
    Div,
    /// Matrix multiplication: a @ b
    MatMul,
    /// Sum reduction along dimensions
    Sum {
        /// Dimensions to reduce over (None = all dimensions)
        dims: Option<Vec<usize>>,
        /// Whether to keep reduced dimensions
        keepdim: bool,
    },
    /// Mean reduction along dimensions
    Mean {
        /// Dimensions to reduce over (None = all dimensions)
        dims: Option<Vec<usize>>,
        /// Whether to keep reduced dimensions
        keepdim: bool,
    },
    /// Element-wise operations
    UnaryOp(UnaryOperation),
    /// Custom operation with backward function
    Custom {
        /// Name of the custom operation for debugging
        name: String,
    },
}

/// Unary operations
#[derive(Debug, Clone)]
pub enum UnaryOperation {
    /// Negation: -x
    Neg,
    /// Square: x²
    Square,
    /// Square root: √x
    Sqrt,
    /// Exponential: e^x
    Exp,
    /// Natural logarithm: ln(x)
    Log,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
}

/// Node in the computation graph
#[derive(Debug)]
pub struct GraphNode {
    /// Unique identifier
    pub id: NodeId,
    /// The operation this node represents
    pub operation: Operation,
    /// Input nodes (dependencies)
    pub inputs: Vec<NodeId>,
    /// Output nodes (dependents)
    pub outputs: Vec<NodeId>,
    /// Node metadata for optimization and debugging
    pub metadata: NodeMetadata,
}

/// Metadata associated with graph nodes
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    /// Human-readable name for debugging
    pub name: String,
    /// Whether this node requires gradients
    pub requires_grad: bool,
    /// Estimated computational cost
    pub cost_estimate: usize,
    /// Memory usage estimate
    pub memory_estimate: usize,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            requires_grad: false,
            cost_estimate: 1,
            memory_estimate: 0,
        }
    }
}

impl GraphNode {
    /// Create a new graph node
    #[must_use]
    pub fn new(id: NodeId, operation: Operation, inputs: Vec<NodeId>) -> Self {
        Self {
            id,
            operation,
            inputs,
            outputs: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Set the node name for debugging
    #[must_use]
    pub fn with_name(mut self, name: String) -> Self {
        self.metadata.name = name;
        self
    }

    /// Mark this node as requiring gradients
    #[must_use]
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.metadata.requires_grad = requires_grad;
        self
    }

    /// Add an output dependency
    pub fn add_output(&mut self, output_id: NodeId) {
        self.outputs.push(output_id);
    }

    /// Check if this node has any inputs
    #[must_use]
    pub fn has_inputs(&self) -> bool {
        !self.inputs.is_empty()
    }

    /// Check if this node produces gradients
    #[must_use]
    pub fn produces_grad(&self) -> bool {
        self.metadata.requires_grad && self.has_inputs()
    }
}

/// Registry for managing graph nodes
#[derive(Debug)]
pub struct NodeRegistry {
    nodes: HashMap<NodeId, Arc<RwLock<GraphNode>>>,
    next_id: usize,
}

impl NodeRegistry {
    /// Create a new empty node registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a new node and return its ID
    pub fn register(&mut self, operation: Operation, inputs: Vec<NodeId>) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = GraphNode::new(id, operation, inputs.clone());
        let node_arc = Arc::new(RwLock::new(node));

        // Update output links for input nodes
        for input_id in inputs {
            if let Some(input_node) = self.nodes.get(&input_id) {
                if let Ok(mut input_node_guard) = input_node.write() {
                    input_node_guard.add_output(id);
                }
            }
        }

        self.nodes.insert(id, node_arc);
        id
    }

    /// Get a node by ID
    #[must_use]
    pub fn get(&self, id: &NodeId) -> Option<Arc<RwLock<GraphNode>>> {
        self.nodes.get(id).cloned()
    }

    /// Get all nodes
    #[must_use]
    pub fn nodes(&self) -> &HashMap<NodeId, Arc<RwLock<GraphNode>>> {
        &self.nodes
    }

    /// Clear all nodes
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_id = 0;
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Topological sorting for backward pass computation order
pub struct TopologicalSorter;

impl TopologicalSorter {
    /// Perform topological sort starting from the given nodes
    /// Returns nodes in reverse topological order (suitable for backward pass)
    #[must_use]
    pub fn sort_from_leaves(registry: &NodeRegistry, leaf_nodes: &[NodeId]) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        let mut visiting = std::collections::HashSet::new();

        for &node_id in leaf_nodes {
            if !visited.contains(&node_id) {
                Self::dfs(registry, node_id, &mut visited, &mut visiting, &mut order);
            }
        }

        // Reverse to get topological order (dependencies first)
        order.reverse();
        order
    }

    /// Depth-first search for topological sorting
    fn dfs(
        registry: &NodeRegistry,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        visiting: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) {
        // Check for cycles
        assert!(
            !visiting.contains(&node_id),
            "Cycle detected in computation graph at node {node_id:?}"
        );

        if visited.contains(&node_id) {
            return;
        }

        visiting.insert(node_id);

        // Visit all dependencies (inputs) first
        if let Some(node_arc) = registry.get(&node_id) {
            if let Ok(node) = node_arc.read() {
                for &input_id in &node.inputs {
                    Self::dfs(registry, input_id, visited, visiting, order);
                }
            }
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        order.push(node_id);
    }

    /// Get all nodes that need gradients (have `requires_grad` = true)
    #[must_use]
    pub fn gradient_nodes(registry: &NodeRegistry) -> Vec<NodeId> {
        registry
            .nodes()
            .iter()
            .filter_map(|(&id, node_arc)| {
                node_arc.read().ok().and_then(
                    |node| {
                        if node.produces_grad() {
                            Some(id)
                        } else {
                            None
                        }
                    },
                )
            })
            .collect()
    }
}

/// Efficient gradient accumulator for autograd operations
///
/// This accumulator provides proper gradient accumulation by:
/// - Storing gradients by tensor identity (pointer-based)
/// - Adding to existing gradients instead of replacing
/// - Memory-efficient storage for multiple backward passes
///
/// Note: Currently restricted to `DenseStorage` as it provides the arithmetic
/// operations needed for gradient accumulation.
pub struct GradientAccumulator<B, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    /// Accumulated gradients for each tensor (keyed by tensor pointer)
    gradients: std::collections::HashMap<*const (), Tensor<B, coeus_storage::DenseStorage<T>, T>>,
}

impl<B, T> GradientAccumulator<B, T>
where
    B: Backend<Data = T> + Default,
    T: DataType + core::ops::Add<Output = T>,
{
    /// Create a new gradient accumulator
    #[must_use]
    pub fn new() -> Self {
        Self {
            gradients: std::collections::HashMap::new(),
        }
    }

    /// Accumulate a gradient for a tensor
    ///
    /// If a gradient already exists for this tensor, the new gradient is added to it.
    /// This enables proper gradient accumulation for tensors that receive gradients
    /// from multiple paths in the computational graph.
    pub fn accumulate(
        &mut self,
        tensor: &Tensor<B, coeus_storage::DenseStorage<T>, T>,
        grad: Tensor<B, coeus_storage::DenseStorage<T>, T>,
    ) {
        let key = (tensor as *const Tensor<B, coeus_storage::DenseStorage<T>, T>).cast::<()>();

        if let Some(existing_grad) = self.gradients.get_mut(&key) {
            // Add the new gradient to the existing one
            // Use the tensor addition operation for proper accumulation
            *existing_grad = &*existing_grad + &grad;
        } else {
            // First gradient for this tensor, just store it
            self.gradients.insert(key, grad);
        }
    }

    /// Get accumulated gradient for a tensor
    #[must_use]
    pub fn get(
        &self,
        tensor: &Tensor<B, coeus_storage::DenseStorage<T>, T>,
    ) -> Option<&Tensor<B, coeus_storage::DenseStorage<T>, T>> {
        let key = (tensor as *const Tensor<B, coeus_storage::DenseStorage<T>, T>).cast::<()>();
        self.gradients.get(&key)
    }

    /// Apply accumulated gradients to tensors
    ///
    /// This method sets the accumulated gradients on their corresponding tensors.
    /// Gradients are properly accumulated within this accumulator before being applied.
    #[allow(clippy::missing_errors_doc)]
    pub fn apply_gradients(&mut self) -> Result<(), crate::error::AutogradError> {
        for (tensor_ptr, grad) in self.gradients.drain() {
            // Convert pointer back to reference (unsafe but controlled)
            let tensor =
                unsafe { &*tensor_ptr.cast::<Tensor<B, coeus_storage::DenseStorage<T>, T>>() };

            // Check if tensor already has a gradient and accumulate
            if let Ok(existing_grad) = tensor.grad() {
                // Accumulate gradients manually
                let existing_data = existing_grad.as_slice();
                let grad_data = grad.as_slice();
                let mut accumulated_data = Vec::with_capacity(existing_data.len());

                for (a, b) in existing_data.iter().zip(grad_data) {
                    accumulated_data.push(*a + *b);
                }

                let accumulated: Tensor<B, DenseStorage<T>, T> = Tensor::from_vec(accumulated_data, existing_grad.shape().dims())
                    .map_err(crate::error::AutogradError::TensorError)?;

                if let Err(e) = tensor.set_grad(accumulated) {
                    return Err(crate::error::AutogradError::GradientError {
                        message: format!("Failed to set accumulated gradient: {e:?}"),
                    });
                }
            } else {
                // No existing gradient, just set it
                if let Err(e) = tensor.set_grad(grad) {
                    return Err(crate::error::AutogradError::GradientError {
                        message: format!("Failed to set gradient: {e:?}"),
                    });
                }
            }
        }

        Ok(())
    }

    /// Clear all accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl<B, T> Default for GradientAccumulator<B, T>
where
    B: Backend<Data = T> + Default,
    T: DataType + core::ops::Add<Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}
