//! PyTorch-compatible automatic differentiation system
//!
//! This module provides gradient computation through automatic graph traversal,
//! compatible with `PyTorch`'s dynamic graph construction and backward pass.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};

use crate::error::{AutogradError, Result};

/// Gradient computation engine for automatic differentiation
///
/// Provides PyTorch-compatible backward pass through automatic graph traversal.
/// Unlike the abandoned node-based approach, this uses Function objects attached
/// to tensors via `grad_fn` for memory-efficient gradient computation.
#[derive(Debug, Default)]
pub struct GradientEngine {
    /// Set of visited functions during backward pass to prevent cycles
    visited: HashSet<usize>,
}

/// Node in the computation graph for topological sorting
#[derive(Debug)]
struct GraphNode<B, S, T>
where
    B: Backend<Data = T> + 'static,
    S: Storage<T> + 'static,
    T: DataType,
{
    /// The function this node represents
    #[allow(dead_code)]
    function: Arc<dyn tensor::Function<B, S, T>>,
    /// Incoming edges (functions that depend on this one)
    incoming: Vec<usize>,
    /// Outgoing edges (functions this one depends on)
    outgoing: Vec<usize>,
    /// Indegree for topological sorting
    indegree: usize,
}

/// Computation graph for topological sorting
#[derive(Debug)]
struct ComputationGraph<B, S, T>
where
    B: Backend<Data = T> + 'static,
    S: Storage<T> + 'static,
    T: DataType,
{
    /// All nodes in the graph
    nodes: Vec<GraphNode<B, S, T>>,
    /// Map from function pointer to node index
    function_to_index: HashMap<usize, usize>,
}

impl GradientEngine {
    /// Create a new gradient computation engine
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build computation graph starting from a root tensor
    #[allow(dead_code)]
    fn build_computation_graph<B, S, T>(
        root_tensor: &tensor::Tensor<B, S, T>,
    ) -> Result<ComputationGraph<B, S, T>>
    where
        B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T>
            + core::fmt::Debug
            + Send
            + Sync
            + 'static
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>
            + StorageFromVec<T>,
        T: DataType,
    {
        let mut graph = ComputationGraph {
            nodes: Vec::new(),
            function_to_index: HashMap::new(),
        };

        // Start from root tensor and traverse backward
        if let Some(root_grad_fn) = root_tensor.grad_fn() {
            Self::build_graph_recursive(&mut graph, root_grad_fn)?;
        }

        Ok(graph)
    }

    /// Recursively build the computation graph
    #[allow(dead_code)]
    fn build_graph_recursive<B, S, T>(
        graph: &mut ComputationGraph<B, S, T>,
        function: &Arc<dyn tensor::Function<B, S, T>>,
    ) -> Result<usize>
    where
        B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T>
            + core::fmt::Debug
            + Send
            + Sync
            + 'static
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>
            + StorageFromVec<T>,
        T: DataType,
    {
        let function_ptr = Arc::as_ptr(function);
        let function_id = function_ptr.cast::<()>() as usize;

        // Check if we've already processed this function
        if let Some(&index) = graph.function_to_index.get(&function_id) {
            return Ok(index);
        }

        // Create new node with placeholder values
        let node_index = graph.nodes.len();
        graph.function_to_index.insert(function_id, node_index);

        // Push a placeholder node so the index is reserved and valid
        graph.nodes.push(GraphNode {
            function: Arc::clone(function),
            incoming: Vec::new(),
            outgoing: Vec::new(),
            indegree: 0,
        });

        // Process input tensors to find parent functions
        let inputs = function.inputs();
        for input_tensor in inputs {
            if let Some(parent_grad_fn) = input_tensor.grad_fn() {
                // Recursively process parent function
                let parent_index = Self::build_graph_recursive(graph, parent_grad_fn)?;

                // Add edge: parent -> current
                graph.nodes[parent_index].outgoing.push(node_index);
                graph.nodes[node_index].incoming.push(parent_index);
                graph.nodes[node_index].indegree += 1;
            }
        }

        Ok(node_index)
    }

    /// Perform topological sort using Kahn's algorithm
    #[allow(dead_code)]
    fn topological_sort<B, S, T>(graph: &ComputationGraph<B, S, T>) -> Result<Vec<usize>>
    where
        B: Backend<Data = T> + 'static,
        S: Storage<T> + 'static,
        T: DataType,
    {
        let mut indegree = graph
            .nodes
            .iter()
            .map(|node| node.indegree)
            .collect::<Vec<_>>();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Find all nodes with indegree 0 (leaf nodes)
        for (i, &deg) in indegree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        while let Some(node_index) = queue.pop_front() {
            result.push(node_index);

            // Reduce indegree of neighbors
            for &neighbor in &graph.nodes[node_index].outgoing {
                indegree[neighbor] -= 1;
                if indegree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // Check for cycles
        if result.len() != graph.nodes.len() {
            return Err(AutogradError::GraphError(
                "Computation graph contains cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Compute gradients through automatic graph traversal
    ///
    /// This implements PyTorch-compatible backward pass by traversing the `grad_fn` chain
    /// using topological sorting to handle shared nodes and complex graph structures correctly.
    ///
    /// # Arguments
    /// * `root_grad_fn` - The `grad_fn` of the tensor to start backward pass from
    /// * `grad_output` - Initial gradient w.r.t. the output tensor
    ///
    /// # Errors
    /// Returns error if backward pass fails
    pub fn backward<B, S, T>(
        &mut self,
        root_grad_fn: Option<&Arc<dyn tensor::Function<B, S, T>>>,
        grad_output: &tensor::Tensor<B, S, T>,
        create_graph: bool,
    ) -> Result<()>
    where
        B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
        S: Storage<T>
            + core::fmt::Debug
            + Send
            + Sync
            + 'static
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>
            + StorageFromVec<T>
            + Clone,
        T: DataType + num_traits::Zero + Clone + 'static + std::ops::Neg<Output = T>,
    {
        let Some(root_fn) = root_grad_fn else {
            return Ok(());
        };

        self.visited.clear();

        let prev_grad_enabled = tensor::tensor_core::grad_enabled();
        tensor::tensor_core::set_grad_enabled(create_graph);

        // 1. Build the computation graph
        let mut graph = ComputationGraph {
            nodes: Vec::new(),
            function_to_index: HashMap::new(),
        };
        Self::build_graph_recursive(&mut graph, root_fn)?;

        // 2. Perform topological sort
        let sorted_indices = Self::topological_sort(&graph)?;

        // 3. Initialize accumulated gradients map
        // Maps function pointer (id) to its accumulated gradient
        let mut accumulated_grads: HashMap<usize, tensor::Tensor<B, storage::DenseStorage<T>, T>> =
            HashMap::new();

        // Set initial gradient for the root function
        let root_id = Arc::as_ptr(root_fn).cast::<()>() as usize;
        let grad_output_dense = grad_output
            .to_dense_generic()
            .map_err(AutogradError::from)?;

        // If create_graph is true, the initial gradient should also track gradients
        let grad_output_dense = if create_graph {
            grad_output_dense.requires_grad_(true)
        } else {
            grad_output_dense
        };

        accumulated_grads.insert(root_id, grad_output_dense);

        // 4. Process nodes in reverse topological order (from outputs to inputs)
        for &node_idx in sorted_indices.iter().rev() {
            let node = &graph.nodes[node_idx];
            let function_id = Arc::as_ptr(&node.function).cast::<()>() as usize;

            // Get accumulated gradient for this function
            let Some(grad_out) = accumulated_grads.remove(&function_id) else {
                continue; // No gradient reached this node
            };

            if !self.visited.insert(function_id) {
                continue;
            }

            // Call backward on the function
            let input_gradients =
                node.function
                    .backward(&grad_out)
                    .map_err(|e| AutogradError::InvalidOperation {
                        operation: format!(
                            "Function {} backward failed: {e}",
                            node.function.name()
                        ),
                    })?;

            // Accumulate gradients into inputs
            let inputs = node.function.inputs();
            for (input_tensor, grad_in_dense) in inputs.iter().zip(input_gradients) {
                // Convert gradient to dense for accumulation if it's not already dense
                let grad_in_dense_converted = grad_in_dense
                    .to_dense_generic()
                    .map_err(AutogradError::from)?;

                // If input has a grad_fn, accumulate into the map for the next layer
                if let Some(parent_fn) = input_tensor.grad_fn() {
                    let parent_id = Arc::as_ptr(parent_fn).cast::<()>() as usize;

                    if let Some(existing_grad) = accumulated_grads.get_mut(&parent_id) {
                        // Use autograd-aware add if create_graph is true
                        let updated_grad = if create_graph {
                            crate::tensor_ops::add(existing_grad, &grad_in_dense_converted)?
                        } else {
                            tensor::ops::arithmetic::add(existing_grad, &grad_in_dense_converted)
                                .map_err(AutogradError::TensorError)?
                        };
                        *existing_grad = updated_grad;
                    } else {
                        accumulated_grads.insert(parent_id, grad_in_dense_converted.clone());
                    }
                }

                // If input requires grad, also accumulate into its .grad field
                if input_tensor.requires_grad() {
                    Self::accumulate_gradient(
                        input_tensor,
                        &grad_in_dense_converted,
                        create_graph,
                    )?;
                }
            }
        }

        // Restore grad enabled state
        tensor::tensor_core::set_grad_enabled(prev_grad_enabled);

        Ok(())
    }

    /// Backward pass for a single tensor
    #[allow(clippy::missing_errors_doc)]
    pub fn backward_tensor<B, S, T>(tensor: &tensor::Tensor<B, S, T>) -> Result<()>
    where
        B: Backend<Data = T> + core::fmt::Debug + Send + Sync + 'static + Clone + Default,
        S: Storage<T>
            + core::fmt::Debug
            + Send
            + Sync
            + 'static
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>
            + StorageFromVec<T>
            + Clone,
        T: DataType + num_traits::Zero + Clone + 'static + std::ops::Neg<Output = T>,
    {
        let Some(grad_fn) = tensor.grad_fn() else {
            return Ok(());
        };

        // Create initial gradient of ones with same shape as tensor
        let shape = tensor.shape();
        let grad_data = vec![T::one(); shape.size()];
        let grad_output = tensor::Tensor::<B, S, T>::from_vec_with_backend(
            grad_data,
            shape.dims(),
            tensor.backend().clone(),
        )
        .map_err(AutogradError::TensorError)?;

        let mut engine = GradientEngine::new();
        engine.backward(Some(grad_fn), &grad_output, false)?;
        Ok(())
    }

    /// Accumulate gradient into a tensor's grad field
    #[allow(clippy::used_underscore_binding)]
    fn accumulate_gradient<B, S, GS, T>(
        tensor: &tensor::Tensor<B, S, T>,
        gradient: &tensor::Tensor<B, GS, T>,
        create_graph: bool,
    ) -> Result<()>
    where
        B: Backend<Data = T> + std::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T>
            + std::fmt::Debug
            + Send
            + Sync
            + Clone
            + 'static
            + StorageFromVec<T>
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>,
        GS: Storage<T> + StorageToDense<T> + StorageFromVec<T> + 'static + tensor::ops::TensorStorageOps<T>,
        T: DataType + Clone + std::ops::Neg<Output = T>,
    {
        if create_graph {
            let gradient_dense = gradient.to_dense_generic().map_err(AutogradError::from)?;

            match tensor.grad() {
                Ok(existing_dense) => {
                    let updated = crate::tensor_ops::add(&existing_dense, &gradient_dense)?;
                    tensor.set_grad(updated).map_err(AutogradError::TensorError)
                }
                Err(_) => tensor
                    .set_grad(gradient_dense)
                    .map_err(AutogradError::TensorError),
            }
        } else {
            tensor
                .accumulate_grad(gradient)
                .map_err(AutogradError::TensorError)
        }
    }

    /// Get accumulated gradient for a tensor (always returns dense tensor)
    #[allow(dead_code)]
    fn get_accumulated_gradient<B, S, T>(
        tensor: &tensor::Tensor<B, S, T>,
    ) -> Result<tensor::Tensor<B, storage::DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + std::fmt::Debug + Send + Sync + 'static + Clone,
        S: Storage<T>
            + std::fmt::Debug
            + Send
            + Sync
            + 'static
            + Clone
            + StorageFromVec<T>
            + StorageToDense<T>
            + tensor::ops::TensorStorageOps<T>,
        T: DataType,
    {
        tensor.grad().map_err(AutogradError::TensorError)
    }
}

/// Perform backward pass on a tensor
///
/// # Arguments
/// * `tensor` - Tensor to compute gradients for
///
/// # Errors
/// Returns error if backward pass fails
pub fn backward<B, S, T>(tensor: &tensor::Tensor<B, S, T>) -> Result<()>
where
    B: Backend<Data = T> + std::fmt::Debug + Send + Sync + Clone + 'static,
    S: Storage<T>
        + std::fmt::Debug
        + Send
        + Sync
        + Clone
        + 'static
        + StorageToDense<T>
        + tensor::ops::TensorStorageOps<T>
        + storage::StorageFromVec<T>,
    T: DataType + std::ops::Neg<Output = T>,
{
    let mut engine = GradientEngine::new();
    if let Some(grad_fn) = tensor.grad_fn() {
        let one_storage = S::from_vec(vec![T::one()], &[])
            .map_err(|e| AutogradError::TensorError(tensor::TensorError::StorageError(e)))?;
        let grad_output = tensor::Tensor::from_storage(one_storage, tensor.backend().clone());

        engine.backward(Some(grad_fn), &grad_output, false)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_gradient_engine_creation() {
        let engine = GradientEngine::new();
        assert!(engine.visited.is_empty());
    }

    #[test]
    fn test_backward_with_none_grad_fn() -> Result<()> {
        let mut engine = GradientEngine::new();
        let grad_tensor = tensor::Tensor::<
            backend::CpuBackend<Float32>,
            storage::DenseStorage<Float32>,
            Float32,
        >::from_vec(vec![Float32::new(1.0)], &[1])?;
        let result = engine.backward(None, &grad_tensor, false);
        assert!(result.is_ok());
        Ok(())
    }
}
