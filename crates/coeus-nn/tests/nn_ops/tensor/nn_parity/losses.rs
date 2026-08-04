use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::{BackendError, SequentialBackend};
use coeus_tensor::Tensor as CoeusTensor;

#[test]
fn test_softmax_parity() {
    let x_data = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 2.5]; // shape [2, 3]

    // Coeus setup
    let x_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &x_data),
        true,
    );
    let out_coeus = coeus_nn::softmax(&x_coeus, 1);

    // Verify forward
    let expected_softmax_out = vec![
        0.090031f32,
        0.244728f32,
        0.665241f32,
        0.025909f32,
        0.116115f32,
        0.857977f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_softmax_out, 1e-4);

    // Backward
    let loss_coeus = coeus_autograd::sum(&out_coeus);
    loss_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let dx_coeus = x_coeus.grad().unwrap();
    let expected_softmax_dx = vec![
        0.000000f32,
        0.000000f32,
        0.000000f32,
        0.000000f32,
        0.000000f32,
        0.000000f32,
    ];

    assert_tensor_eq_data::<SequentialBackend>(&dx_coeus, &expected_softmax_dx, 1e-4);
}

#[test]
fn test_cross_entropy_loss_parity() {
    let logits_data = vec![1.5f32, 0.5, -0.5, -1.0, 2.0, 0.0]; // shape [2, 3]
    let targets_data = vec![0, 1]; // batch size 2

    // Coeus setup
    let logits_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &logits_data),
        true,
    );
    let loss_coeus = coeus_nn::cross_entropy_loss(&logits_coeus, &targets_data)
        .expect("invariant: parity inputs have valid cross-entropy shapes and targets");

    // Verify forward (Mean Cross Entropy: mean(-log(softmax(logits)[target])))
    let expected_cross_entropy_out = vec![0.288726f32];
    assert_tensor_eq_data::<SequentialBackend>(
        &loss_coeus.tensor,
        &expected_cross_entropy_out,
        1e-4,
    );

    // Backward
    loss_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let dlogits_coeus = logits_coeus.grad().unwrap();
    let expected_cross_entropy_dlogits = vec![
        -0.167379f32,
        0.122364f32,
        0.045015f32,
        0.021005f32,
        -0.078103f32,
        0.057098f32,
    ];

    assert_tensor_eq_data::<SequentialBackend>(
        &dlogits_coeus,
        &expected_cross_entropy_dlogits,
        1e-4,
    );
}

#[test]
fn cross_entropy_rejects_invalid_contracts_before_autograd_registration() {
    let rank_one = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice([3], &[1.0, 2.0, 3.0]),
        true,
    );
    assert!(matches!(
        coeus_nn::cross_entropy_loss(&rank_one, &[0]),
        Err(BackendError::UnsupportedRank { rank: 1, .. })
    ));

    let logits = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        true,
    );
    assert!(matches!(
        coeus_nn::cross_entropy_loss(&logits, &[0]),
        Err(BackendError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        coeus_nn::cross_entropy_loss(&logits, &[0, 3]),
        Err(BackendError::IndexOutOfRange {
            position: 1,
            index: 3,
            bound: 3,
            ..
        })
    ));

    let empty_classes = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice([2, 0], &[]),
        true,
    );
    assert!(matches!(
        coeus_nn::cross_entropy_loss(&empty_classes, &[0, 0]),
        Err(BackendError::EmptyDimension {
            dimension: "class",
            ..
        })
    ));
}
