use coeus_autograd::{add, no_grad_guard, relu, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn no_grad_blocks_operation_graph_construction() {
    let backend = MoiraiBackend::new();
    let x = Var::new(
        Tensor::from_slice_on(vec![3], &[1.0f64, -2.0, 3.0], &backend),
        true,
    );

    {
        let _guard = no_grad_guard();
        let y = relu(&x);
        assert_eq!(y.tensor.as_slice(), &[1.0, 0.0, 3.0]);
        assert!(y.grad.is_none(), "no_grad op output must not allocate grad");
        assert!(
            y.creator.is_none(),
            "no_grad op output must not allocate a backward node"
        );
    }

    let tracked = relu(&x);
    // The guard's counterpart to the two `is_none` assertions above: a
    // resumed op allocates a zeroed accumulator of the output's shape, not
    // merely a slot. `is_some` could not tell the two apart.
    assert_eq!(
        tracked
            .grad()
            .expect("tracking must resume after guard drop")
            .as_slice(),
        &[0.0, 0.0, 0.0],
        "a resumed op's accumulator starts zeroed at the output's shape"
    );
    // The backward node's presence is not asserted structurally: the
    // gradient reaching `x` three lines below is what a node existing and
    // propagating actually means, and `is_some` restates it more weakly.
    sum(&tracked)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(x.grad().unwrap().as_slice(), &[1.0, 0.0, 1.0]);
}

#[test]
fn no_grad_preserves_explicit_leaf_requires_grad() {
    let backend = MoiraiBackend::new();
    let x = {
        let _guard = no_grad_guard();
        Var::new(
            Tensor::from_slice_on(vec![2], &[2.0f64, 4.0], &backend),
            true,
        )
    };

    assert_eq!(
        x.grad()
            .expect("explicit leaf requires_grad must be honored")
            .as_slice(),
        &[0.0, 0.0],
        "an explicitly tracked leaf starts with a zeroed accumulator"
    );

    let y = add(&x, &x);
    assert_eq!(y.tensor.as_slice(), &[4.0, 8.0]);
    assert_eq!(
        y.grad()
            .expect("tracking must resume for later operations")
            .as_slice(),
        &[0.0, 0.0],
        "a post-guard operation starts with a zeroed accumulator"
    );
    sum(&y)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(x.grad().unwrap().as_slice(), &[2.0, 2.0]);
}
