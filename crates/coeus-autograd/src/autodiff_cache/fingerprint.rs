//! Computation-graph fingerprint for [`super::ComputeGraphCache`] keys.

/// Compute a fingerprint for a computation graph.
///
/// This function computes a hash of the graph structure based on:
/// - Input tensor shapes
/// - Operation sequence
/// - Backend type identifier
///
/// The fingerprint can be used as a cache key to identify repeated patterns.
pub fn compute_graph_fingerprint(
    op_names: &[&str],
    input_shapes: &[&[usize]],
    backend_id: u32,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash operation sequence
    for op in op_names {
        op.hash(&mut hasher);
    }

    // Hash input shapes
    for shape in input_shapes {
        for dim in *shape {
            dim.hash(&mut hasher);
        }
        // Add shape length to distinguish [2,3] from [2],[3]
        shape.len().hash(&mut hasher);
    }

    // Hash backend type
    backend_id.hash(&mut hasher);

    hasher.finish()
}
