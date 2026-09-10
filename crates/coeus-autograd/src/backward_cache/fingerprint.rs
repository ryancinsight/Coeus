//! Structural graph fingerprints and traversal metadata.

use crate::autodiff_cache::GraphInfo;
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

/// Compute a fingerprint of a computation graph rooted at a given node.
///
/// The fingerprint includes operation names, input shapes, and creator/leaf
/// connectivity. The accompanying metadata describes the graph for cache
/// accounting and diagnostics.
pub fn compute_graph_structure_fingerprint<T: Scalar, B: ComputeBackend + Default>(
    root_node: &Arc<dyn BackwardNode<T, B>>,
) -> (u64, GraphInfo) {
    let (fingerprint, graph_info, _) = collect_graph(root_node, false);
    (fingerprint, graph_info)
}

/// A graph's structural hash, metadata, and optional reverse-mode order.
pub(crate) type CollectedGraph<T, B> = (u64, GraphInfo, Option<Vec<Arc<dyn BackwardNode<T, B>>>>);

/// Collect graph metadata and an optional live post-order in one traversal.
pub(crate) fn collect_graph<T: Scalar, B: ComputeBackend + Default>(
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
/// Keeping traversal state together avoids threading a growing list of
/// mutable parameters through recursive visits.
struct Traversal<T: Scalar, B: ComputeBackend + Default> {
    /// Every node discovered so far, keyed by identity, in discovery order.
    ///
    /// The ordinal makes repeated edges visible to the hash: a second edge to
    /// an already visited creator returns immediately, while its ordinal still
    /// names the node reached by that edge.
    ordinals: HashMap<*const (), usize>,
    hasher: DefaultHasher,
    op_sequence: Vec<String>,
    order: Option<Vec<Arc<dyn BackwardNode<T, B>>>>,
    leaf_count: usize,
    max_depth: usize,
}

impl<T: Scalar, B: ComputeBackend + Default> Traversal<T, B> {
    /// Visit `node` and each reachable node once, returning its ordinal.
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

        // Reverse-mode propagation consumes nodes in post-order.
        if let Some(order) = &mut self.order {
            order.push(node.clone());
        }

        ordinal
    }

    /// Consume traversal state into its fingerprint, metadata, and order.
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
