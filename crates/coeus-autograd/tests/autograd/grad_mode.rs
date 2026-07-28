use coeus_autograd::{add, no_grad_guard, relu, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn no_grad_blocks_operation_graph_construction() {
    let backend = MoiraiBackend::new();
    let x = Var::new(
        Tensor::from_slice_on(vec![3], &[1.0f64, -2.0, 3.0], &backend).expect("valid tensor construction"),
        true,
    ).expect("valid variable construction");

    {
        let _guard = no_grad_guard();
        let y = relu(&x).expect("valid autograd operation");
        assert_eq!(y.tensor.as_slice(), &[1.0, 0.0, 3.0]);
        assert!(y.grad.is_none(), "no_grad op output must not allocate grad");
        assert!(
            y.creator.is_none(),
            "no_grad op output must not allocate a backward node"
        );
    }

    let tracked = relu(&x).expect("valid autograd operation");
    assert!(
        tracked.grad.is_some(),
        "tracking must resume after guard drop"
    );
    assert!(
        tracked.creator.is_some(),
        "tracked op output must carry a backward node"
    );

    sum(&tracked)
        .expect("valid autograd operation")
        .backward()
        .expect("valid backward propagation");
    assert_eq!(x.grad().unwrap().as_slice(), &[1.0, 0.0, 1.0]);
}

#[test]
fn no_grad_preserves_explicit_leaf_requires_grad() {
    let backend = MoiraiBackend::new();
    let x = {
        let _guard = no_grad_guard();
        Var::new(
            Tensor::from_slice_on(vec![2], &[2.0f64, 4.0], &backend).expect("valid tensor construction"),
            true,
        ).expect("valid variable construction")
    };

    assert!(
        x.grad.is_some(),
        "explicit leaf requires_grad must be honored"
    );

    let y = add(&x, &x).expect("valid autograd operation");
    assert_eq!(y.tensor.as_slice(), &[4.0, 8.0]);
    assert!(
        y.grad.is_some(),
        "tracking must resume for later operations"
    );
    sum(&y)
        .expect("valid autograd operation")
        .backward()
        .expect("valid backward propagation");
    assert_eq!(x.grad().unwrap().as_slice(), &[2.0, 2.0]);
}
