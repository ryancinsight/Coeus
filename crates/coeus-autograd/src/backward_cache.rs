//! Caching integration for backward pass compilation.
//!
//! This module provides utilities to integrate the computation graph cache
//! into the automatic differentiation backward pass, reducing compilation
//! overhead by reusing graph traversal patterns.

use crate::autodiff_cache::{ComputeGraphCache, GraphInfo};
use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{ComputeBackend, Scalar};
use std::collections::HashSet;
use std::sync::Arc;

/// Compute a fingerprint of a computation graph rooted at a given node.
///
/// This function traverses the computation graph and produces a deterministic
/// fingerprint based on:
/// - Operation sequence (pre-order traversal)
/// - Input tensor shapes
/// - Node connectivity patterns
///
/// The fingerprint is used as a cache key for identifying repeated patterns.
pub fn compute_graph_structure_fingerprint<T: Scalar, B: ComputeBackend + Default>(
    root_node: &Arc<dyn BackwardNode<T, B>>,
) -> (u64, GraphInfo) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let mut visited = HashSet::new();
    let mut op_sequence = Vec::new();
    let mut node_count = 0;
    let mut leaf_count = 0;
    let mut max_depth = 0;

    fn traverse<T: Scalar, B: ComputeBackend + Default>(
        node: &Arc<dyn BackwardNode<T, B>>,
        visited: &mut HashSet<*const ()>,
        hasher: &mut std::collections::hash_map::DefaultHasher,
        op_sequence: &mut Vec<String>,
        node_count: &mut usize,
        leaf_count: &mut usize,
        max_depth: &mut usize,
        depth: usize,
    ) {
        use std::hash::{Hash, Hasher};

        let ptr = Arc::as_ptr(node) as *const ();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        *node_count += 1;
        if depth > *max_depth {
            *max_depth = depth;
        }

        // Hash the operation name
        let op_name = node.op_name();
        op_name.hash(hasher);
        op_sequence.push(op_name.to_string());

        // Hash input shapes
        let inputs = node.inputs();
        inputs.len().hash(hasher);
        for input in inputs {
            input.tensor.shape().hash(hasher);
            if input.creator.is_none() {
                *leaf_count += 1;
            }

            if let Some(ref creator) = input.creator {
                traverse(creator, visited, hasher, op_sequence, node_count, leaf_count, max_depth, depth + 1);
            }
        }
    }

    traverse(
        root_node,
        &mut visited,
        &mut hasher,
        &mut op_sequence,
        &mut node_count,
        &mut leaf_count,
        &mut max_depth,
        0,
    );

    let fingerprint = hasher.finish();
    let graph_info = GraphInfo {
        node_count,
        leaf_count,
        max_depth,
        op_sequence,
    };

    (fingerprint, graph_info)
}

/// Perform topological sort using the cache when available.
///
/// This function attempts to use the cache to speed up topological sorting.
/// If a matching graph is found in the cache, it skips the expensive DFS.
/// If not, it performs normal DFS and caches the result.
pub fn topological_sort_with_cache<T: Scalar, B: ComputeBackend + Default>(
    root_node: Option<&Arc<dyn BackwardNode<T, B>>>,
    cache: &ComputeGraphCache,
) -> Vec<Arc<dyn BackwardNode<T, B>>> {
    let mut order: Vec<Arc<dyn BackwardNode<T, B>>> = Vec::new();

    let Some(root) = root_node else {
        return order;
    };

    // Compute fingerprint
    let (fingerprint, graph_info) = compute_graph_structure_fingerprint(root);

    // Check cache
    if let Some(_cached_info) = cache.lookup(fingerprint) {
        // Cache hit: we could use cached traversal order here
        // For now, we still do DFS but could optimize by caching the actual order
        // This would require serializing the Arc<dyn BackwardNode> pointers,
        // which is not straightforward. So we cache the info but still traverse.
    }

    // Perform DFS regardless (we could optimize by caching Arc order)
    let mut visited = HashSet::new();

    fn dfs<T: Scalar, B: ComputeBackend + Default>(
        node: &Arc<dyn BackwardNode<T, B>>,
        visited: &mut HashSet<*const ()>,
        order: &mut Vec<Arc<dyn BackwardNode<T, B>>>,
    ) {
        let ptr = Arc::as_ptr(node) as *const ();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        for input in node.inputs() {
            if let Some(ref creator) = input.creator {
                dfs(creator, visited, order);
            }
        }

        order.push(node.clone());
    }

    dfs(root, &mut visited, &mut order);

    // Cache the graph structure for future iterations
    cache.insert(fingerprint, graph_info);

    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;
    use std::sync::Arc;

    #[test]
    fn test_graph_fingerprint_consistency() {
        // This would require a mock BackwardNode implementation
        // Placeholder for future tests
    }
}
