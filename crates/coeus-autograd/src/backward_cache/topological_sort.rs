//! Cached topological ordering for reverse-mode propagation.

use super::fingerprint::collect_graph;
use crate::autodiff_cache::ComputeGraphCache;
use crate::node::BackwardNode;
use coeus_core::{ComputeBackend, Scalar};
use std::sync::Arc;

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
