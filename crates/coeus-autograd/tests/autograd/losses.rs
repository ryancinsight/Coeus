use coeus_autograd::{huber_loss, Var};
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

#[test]
fn huber_loss_preserves_multidimensional_gradient_shape_and_values() {
    let input = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([2, 2], &[0.0_f64, 0.0, 3.0, -3.0]),
        true,
    );
    let target = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([2, 2], &[0.0_f64, 1.0, 0.0, 0.0]),
        false,
    );

    let loss = huber_loss(&input, &target, 1.0)
        .expect("invariant: matching non-empty shapes and positive finite delta");
    assert_eq!(loss.tensor.as_slice(), &[1.375]);
    loss.backward()
        .expect("invariant: scalar Huber loss has a valid backward seed");

    let gradient = input
        .grad()
        .expect("tracked Huber-loss input receives a gradient");
    assert_eq!(gradient.shape(), &[2, 2]);
    assert_eq!(gradient.as_slice(), &[0.0, -0.25, 0.25, -0.25]);
}

#[test]
fn huber_loss_rejects_shape_mismatch() {
    let input = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([2], &[0.0, 1.0]),
        true,
    );
    let target = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([1, 2], &[0.0, 1.0]),
        false,
    );

    let error = match huber_loss(&input, &target, 1.0) {
        Ok(_) => panic!("different shapes must not enter Huber-loss indexing"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        coeus_core::BackendError::ShapeMismatch {
            operation: "huber_loss",
            lhs,
            rhs
        } if lhs == [2] && rhs == [1, 2]
    ));
}
