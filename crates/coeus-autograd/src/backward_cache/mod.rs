//! Caching integration for backward pass compilation.
//!
//! This module provides utilities to integrate the computation graph cache
//! into the automatic differentiation backward pass. The helper retains a
//! bounded topology plan for the exact live graph instance, while the metadata
//! cache continues to identify repeated graph patterns across graph instances.

use crate::autodiff_cache::{ComputeGraphCache, GraphInfo};
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;
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
    let (fingerprint, graph_info, _) = collect_graph(root_node, false);
    (fingerprint, graph_info)
}

/// A graph's structural hash, its metadata, and the post-order the caller
/// asked for -- `None` when it did not.
type CollectedGraph<T, B> = (u64, GraphInfo, Option<Vec<Arc<dyn BackwardNode<T, B>>>>);

/// Collect graph metadata and a live post-order in one traversal.
fn collect_graph<T: Scalar, B: ComputeBackend + Default>(
    root_node: &Arc<dyn BackwardNode<T, B>>,
    collect_order: bool,
) -> CollectedGraph<T, B> {
    let mut traversal = Traversal {
        ordinals: HashMap::new(),
        hasher: DefaultHasher::new(),
        op_sequence: Vec::new(),
        order: collect_order.then(Vec::new),
        leaf_count: 0,
        max_depth: 0,
    };
    traversal.visit(root_node, 0);
    traversal.finish()
}

/// Edge tag hashed for an input that has a creator.
const EDGE_CREATOR: u8 = 1;

/// Edge tag hashed for an input that is a leaf.
const EDGE_LEAF: u8 = 2;

/// Everything one traversal accumulates.
///
/// These were eight `&mut` parameters threaded through a recursive function,
/// which meant every call site restated the whole set in order and a new
/// statistic meant touching all of them.
struct Traversal<T: Scalar, B: ComputeBackend + Default> {
    /// Every node discovered so far, keyed by identity, in discovery order.
    ///
    /// The ordinal is what makes a repeated edge visible to the hash: a
    /// second edge to an already-visited creator ends its `visit` call
    /// immediately and so contributes nothing of its own, but its ordinal
    /// still names which node the edge reached. The map's length is the
    /// node count.
    ordinals: HashMap<*const (), usize>,
    hasher: DefaultHasher,
    op_sequence: Vec<String>,
    order: Option<Vec<Arc<dyn BackwardNode<T, B>>>>,
    leaf_count: usize,
    max_depth: usize,
}

impl<T: Scalar, B: ComputeBackend + Default> Traversal<T, B> {
    /// Visit `node` and everything reachable from it, once each, and answer
    /// with `node`'s traversal-local ordinal.
    ///
    /// Every input hashes an edge tag naming whether it has a creator, and a
    /// creator edge additionally hashes the ordinal it reaches. Without
    /// those, `[creator C, creator C]` and `[creator C, leaf]` hash alike
    /// whenever their shapes match -- the second edge's `visit` returns at
    /// once and a leaf recurses into nothing -- while their `leaf_count`
    /// differs, so the metadata cache would answer one graph's lookup with
    /// the other's `GraphInfo`.
    fn visit(&mut self, node: &Arc<dyn BackwardNode<T, B>>, depth: usize) -> usize {
        use std::hash::Hash;

        let ptr = Arc::as_ptr(node) as *const ();
        if let Some(&ordinal) = self.ordinals.get(&ptr) {
            return ordinal;
        }
        let ordinal = self.ordinals.len();
        self.ordinals.insert(ptr, ordinal);

        self.max_depth = self.max_depth.max(depth);

        let op_name = node.op_name();
        op_name.hash(&mut self.hasher);
        self.op_sequence.push(op_name.to_string());

        let inputs = node.inputs();
        inputs.len().hash(&mut self.hasher);
        for input in inputs {
            input.tensor.shape().hash(&mut self.hasher);
            match input.creator {
                Some(ref creator) => {
                    EDGE_CREATOR.hash(&mut self.hasher);
                    let creator_ordinal = self.visit(creator, depth + 1);
                    creator_ordinal.hash(&mut self.hasher);
                }
                None => {
                    EDGE_LEAF.hash(&mut self.hasher);
                    self.leaf_count += 1;
                }
            }
        }

        // Post-order is the order required by reverse-mode propagation.
        if let Some(order) = &mut self.order {
            order.push(node.clone());
        }

        ordinal
    }

    /// Consume the traversal into its fingerprint, metadata and post-order.
    fn finish(self) -> CollectedGraph<T, B> {
        let graph_info = GraphInfo {
            node_count: self.ordinals.len(),
            leaf_count: self.leaf_count,
            max_depth: self.max_depth,
            op_sequence: self.op_sequence,
        };
        (self.hasher.finish(), graph_info, self.order)
    }
}

/// Perform topological sort using the cache when available.
///
/// The first call traverses the graph once to compute metadata and build a
/// live post-order. Later calls for the same live root reuse that post-order;
/// separately constructed graphs still traverse independently, avoiding unsafe
/// pointer reuse across graphs.
pub fn topological_sort_with_cache<T: Scalar, B: ComputeBackend + Default>(
    root_node: Option<&Arc<dyn BackwardNode<T, B>>>,
    cache: &ComputeGraphCache,
) -> Vec<Arc<dyn BackwardNode<T, B>>> {
    let Some(root) = root_node else {
        return Vec::new();
    };

    if let Some(plan) = cache.lookup_plan(root) {
        if !cache.record_lookup(plan.fingerprint) {
            cache.insert(plan.fingerprint, (*plan.graph_info).clone());
        }
        return plan.order;
    }

    let (fingerprint, graph_info, order) = collect_graph(root, true);
    if !cache.record_lookup(fingerprint) {
        cache.insert(fingerprint, graph_info.clone());
    }

    let order = order.expect("order collection is enabled for topological sorting");
    cache.insert_plan(root, fingerprint, graph_info, order.clone());
    order
}

#[cfg(test)]
mod tests;
