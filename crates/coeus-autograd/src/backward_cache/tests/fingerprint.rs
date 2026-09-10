//! The structural fingerprint has to separate graphs the metadata cache keys on.

use super::*;

/// One `TestNode` over the given inputs, boxed as a graph node.
fn node(
    name: &'static str,
    inputs: Vec<Var<f32, MoiraiBackend>>,
) -> Arc<dyn BackwardNode<f32, MoiraiBackend>> {
    Arc::new(TestNode {
        name,
        output_grad: Arc::new(GradBuffer::new(Tensor::zeros([1]))),
        inputs,
    })
}

/// A variable produced by `creator`, so an edge to it is a creator edge.
fn output_of(creator: &Arc<dyn BackwardNode<f32, MoiraiBackend>>) -> Var<f32, MoiraiBackend> {
    Var::with_creator(
        Tensor::zeros([1]),
        Some(Arc::clone(creator.output_grad())),
        Arc::clone(creator),
    )
}

#[test]
fn a_repeated_creator_edge_and_a_leaf_edge_fingerprint_apart() {
    // Both roots take two inputs of shape [1] under the same op names. They
    // differ only in what the second edge reaches: the same child again, or a
    // leaf. The second edge to an already-visited child ends its traversal
    // immediately, and a leaf recurses into nothing, so without an edge tag
    // neither contributes anything and the two hash alike.
    let shared = node("child", vec![Var::new(Tensor::zeros([1]), true)]);
    let repeated = node("root", vec![output_of(&shared), output_of(&shared)]);

    let other = node("child", vec![Var::new(Tensor::zeros([1]), true)]);
    let mixed = node(
        "root",
        vec![output_of(&other), Var::new(Tensor::zeros([1]), true)],
    );

    let (repeated_hash, repeated_info) = compute_graph_structure_fingerprint(&repeated);
    let (mixed_hash, mixed_info) = compute_graph_structure_fingerprint(&mixed);

    // The metadata they would key: distinct, which is why sharing a hash is a
    // wrong answer rather than a collision to tolerate.
    assert_eq!(repeated_info.leaf_count, 1);
    assert_eq!(mixed_info.leaf_count, 2);
    assert_ne!(repeated_hash, mixed_hash);
}

#[test]
fn two_creator_edges_to_one_node_differ_from_edges_to_two() {
    // Same op names, same shapes, same node count on the second graph's own
    // terms -- the distinction is that one child is shared and the other pair
    // is not, which the creator ordinal is what records.
    let shared = node("child", vec![Var::new(Tensor::zeros([1]), true)]);
    let one_child = node("root", vec![output_of(&shared), output_of(&shared)]);

    let first = node("child", vec![Var::new(Tensor::zeros([1]), true)]);
    let second = node("child", vec![Var::new(Tensor::zeros([1]), true)]);
    let two_children = node("root", vec![output_of(&first), output_of(&second)]);

    let (one_hash, one_info) = compute_graph_structure_fingerprint(&one_child);
    let (two_hash, two_info) = compute_graph_structure_fingerprint(&two_children);

    assert_eq!(one_info.node_count, 2);
    assert_eq!(two_info.node_count, 3);
    assert_ne!(one_hash, two_hash);
}

#[test]
fn the_same_graph_fingerprints_identically_twice() {
    // The ordinals are traversal-local, so they must not make the hash depend
    // on anything but structure -- two separate traversals of one graph agree.
    let root = node(
        "root",
        vec![output_of(&node(
            "child",
            vec![Var::new(Tensor::zeros([1]), true)],
        ))],
    );

    let (first, _) = compute_graph_structure_fingerprint(&root);
    let (second, _) = compute_graph_structure_fingerprint(&root);

    assert_eq!(first, second);
}
