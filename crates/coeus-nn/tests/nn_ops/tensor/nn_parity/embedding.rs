use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor as CoeusTensor;

#[test]
fn test_embedding_parity() {
    // Vocabulary size = 5, embedding dim = 4
    let w_data = vec![
        0.1f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
        1.0, 1.1, 1.2, 1.3,
    ];
    let indices_data = vec![1.0f32, 2.0, 0.0, 4.0, 3.0, 1.0]; // shape [2, 3]

    // Coeus setup
    let mut emb_coeus = coeus_nn::Embedding::<f32, SequentialBackend>::new(5, 4);
    emb_coeus.weight = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![5, 4], &w_data),
        true,
    );
    let x_coeus = CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &indices_data);
    let out_coeus = emb_coeus.forward_indices(&x_coeus);

    // Verify forward
    let expected_embedding_out = vec![
        -0.100000f32,
        -0.200000f32,
        -0.300000f32,
        -0.400000f32,
        0.500000f32,
        0.600000f32,
        0.700000f32,
        0.800000f32,
        0.100000f32,
        0.200000f32,
        0.300000f32,
        0.400000f32,
        1.000000f32,
        1.100000f32,
        1.200000f32,
        1.300000f32,
        -0.500000f32,
        -0.600000f32,
        -0.700000f32,
        -0.800000f32,
        -0.100000f32,
        -0.200000f32,
        -0.300000f32,
        -0.400000f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_embedding_out, 1e-4);

    // Backward pass
    let loss_coeus = coeus_autograd::sum(&out_coeus);
    loss_coeus.backward();

    let dw_coeus = emb_coeus.weight.grad().unwrap();
    let expected_embedding_dw = vec![
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        2.000000f32,
        2.000000f32,
        2.000000f32,
        2.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
        1.000000f32,
    ];

    assert_tensor_eq_data(&dw_coeus, &expected_embedding_dw, 1e-4);
}
