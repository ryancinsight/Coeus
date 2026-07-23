#![allow(clippy::excessive_precision)]

use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_nn::{
    Conv1d as CoeusConv1d, Conv2d as CoeusConv2d, Conv3d as CoeusConv3d,
    LayerNorm as CoeusLayerNorm, Linear as CoeusLinear, Module as CoeusModule,
};
use coeus_tensor::Tensor as CoeusTensor;

fn assert_tensor_eq_data<B: coeus_core::ComputeBackend>(
    coeus: &CoeusTensor<f32, B>,
    expected: &[f32],
    tol: f32,
) where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let coeus_slice = coeus.as_slice();
    assert_eq!(coeus_slice.len(), expected.len());
    for (i, (&c, &b)) in coeus_slice.iter().zip(expected.iter()).enumerate() {
        let diff = (c - b).abs();
        assert!(
            diff < tol,
            "Mismatch at index {i}: coeus = {c}, expected = {b} (diff = {diff}, tolerance = {tol})"
        );
    }
}

#[test]
fn test_linear_parity() {
    // Inputs: batch_size=2, in_features=3, out_features=2
    let x_data = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 2.5];
    let w_data = vec![0.5f32, -0.5, 1.0, 0.0, 2.0, -1.0];
    let b_data = vec![0.2f32, -0.1];

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 3], &x_data), true);
    let mut linear_coeus = CoeusLinear::<f32, SequentialBackend>::new(3, 2, true);
    linear_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![2, 3], &w_data), true);
    linear_coeus.bias = Some(CoeusVar::new(
        CoeusTensor::from_slice(vec![2], &b_data),
        true,
    ));

    // Coeus forward
    let out_coeus = linear_coeus.forward(&x_coeus);

    // Verify forward
    let expected_linear_out = vec![2.700000f32, 0.900000f32, 1.950000f32, -1.600000f32];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_linear_out, 1e-4);

    // Coeus backward
    out_coeus.backward();

    // Verify gradients
    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = linear_coeus.weight.grad().unwrap();
    let db_coeus = linear_coeus.bias.as_ref().unwrap().grad().unwrap();

    let expected_linear_dx = vec![
        0.500000f32,
        1.500000f32,
        0.000000f32,
        0.500000f32,
        1.500000f32,
        0.000000f32,
    ];
    let expected_linear_dw = vec![
        0.000000f32,
        2.500000f32,
        5.500000f32,
        0.000000f32,
        2.500000f32,
        5.500000f32,
    ];
    let expected_linear_db = vec![2.000000f32, 2.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_linear_dx, 1e-4);
    assert_tensor_eq_data(&dw_coeus, &expected_linear_dw, 1e-4);
    assert_tensor_eq_data(&db_coeus, &expected_linear_db, 1e-4);
}

#[test]
fn test_layernorm_parity() {
    // Inputs: batch_size=2, features=4
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0];
    let w_data = vec![1.2f32, 0.8, 1.0, 0.9];
    let b_data = vec![0.1f32, -0.1, 0.2, 0.0];
    let eps = 1e-5f64;

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 4], &x_data), true);
    let mut ln_coeus = CoeusLayerNorm::<f32, SequentialBackend>::new(4, eps);
    ln_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![4], &w_data), true);
    ln_coeus.bias = CoeusVar::new(CoeusTensor::from_slice(vec![4], &b_data), true);

    // Coeus forward
    let out_coeus = ln_coeus.forward(&x_coeus);

    // Verify forward
    let expected_layernorm_out = vec![
        -1.509963f32,
        -0.457769f32,
        0.647212f32,
        1.207472f32,
        -1.586673f32,
        -0.474816f32,
        0.980867f32,
        0.983893f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_layernorm_out, 1e-3);

    // Coeus backward
    out_coeus.backward();

    // Verify gradients
    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = ln_coeus.weight.grad().unwrap();
    let db_coeus = ln_coeus.bias.grad().unwrap();

    let expected_layernorm_dx = vec![
        0.107332f32,
        -0.187829f32,
        0.053665f32,
        0.026832f32,
        0.075421f32,
        -0.131033f32,
        0.051804f32,
        0.003809f32,
    ];
    let expected_layernorm_dw = vec![-2.747197f32, -0.915732f32, 1.228079f32, 2.434850f32];
    let expected_layernorm_db = vec![2.000000f32, 2.000000f32, 2.000000f32, 2.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_layernorm_dx, 1e-3);
    assert_tensor_eq_data(&dw_coeus, &expected_layernorm_dw, 1e-3);
    assert_tensor_eq_data(&db_coeus, &expected_layernorm_db, 1e-3);
}

#[test]
fn test_conv1d_parity() {
    // Conv1D: batch=1, in_channels=2, length=4, out_channels=3, kernel=3
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
    let w_data = vec![
        0.5f32, -0.5, 1.0, 0.0, 1.0, 0.0, 0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3, 1.0f32, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ];
    let b_data = vec![0.1f32, -0.1, 0.5];

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![1, 2, 4], &x_data), true);
    let mut conv_coeus = CoeusConv1d::<f32, SequentialBackend>::with_params(2, 3, 3, 1, 0, 1, true);
    conv_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![3, 2, 3], &w_data), true);
    conv_coeus.bias = Some(CoeusVar::new(
        CoeusTensor::from_slice(vec![3], &b_data),
        true,
    ));

    // Coeus forward
    let out_coeus = conv_coeus.forward(&x_coeus);

    // Verify forward
    let expected_conv1d_out = vec![
        2.600000f32,
        4.600000f32,
        1.100000f32,
        1.100000f32,
        6.500000f32,
        12.500000f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_conv1d_out, 1e-4);

    // Coeus backward
    out_coeus.backward();

    // Verify gradients
    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = conv_coeus.weight.grad().unwrap();
    let db_coeus = conv_coeus.bias.as_ref().unwrap().grad().unwrap();

    let expected_conv1d_dx = vec![
        1.600000f32,
        2.300000f32,
        3.000000f32,
        2.300000f32,
        0.900000f32,
        2.700000f32,
        2.500000f32,
        0.700000f32,
    ];
    let expected_conv1d_dw = vec![
        3.000000f32,
        5.000000f32,
        7.000000f32,
        -1.000000f32,
        1.000000f32,
        3.000000f32,
        3.000000f32,
        5.000000f32,
        7.000000f32,
        -1.000000f32,
        1.000000f32,
        3.000000f32,
        3.000000f32,
        5.000000f32,
        7.000000f32,
        -1.000000f32,
        1.000000f32,
        3.000000f32,
    ];
    let expected_conv1d_db = vec![2.000000f32, 2.000000f32, 2.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_conv1d_dx, 1e-4);
    assert_tensor_eq_data(&dw_coeus, &expected_conv1d_dw, 1e-4);
    assert_tensor_eq_data(&db_coeus, &expected_conv1d_db, 1e-4);
}

#[test]
fn test_dropout_parity() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let p = 0.0;

    // Coeus setup
    let x_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &x_data),
        true,
    );
    let mut dropout_coeus = coeus_nn::Dropout::new(p);
    dropout_coeus.is_training = true;
    let out_coeus = dropout_coeus.forward(&x_coeus);

    // Verify forward (with p=0, output is equal to input)
    assert_tensor_eq_data(&out_coeus.tensor, &x_data, 1e-4);

    // Backward
    out_coeus.backward();

    let dx_coeus = x_coeus.grad().unwrap();

    // Verify backward (with p=0, output gradient is 1.0 for each element, so sum loss gradient is 1.0 for each)
    let expected_dx = vec![1.0f32; 6];
    assert_tensor_eq_data(&dx_coeus, &expected_dx, 1e-4);

    // Stochastic scaling test in Coeus
    let x_coeus2 = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1000], &vec![1.0f32; 1000]),
        true,
    );
    let mut dropout_coeus2 = coeus_nn::Dropout::new(0.5);
    dropout_coeus2.is_training = true;
    let out_coeus2 = dropout_coeus2.forward(&x_coeus2);
    out_coeus2.backward();

    let slice: &[f32] = out_coeus2.tensor.as_slice();
    for &val in slice {
        assert!(val == 0.0 || (val - 2.0f32).abs() < 1e-5);
    }
    let grad_coeus2 = x_coeus2.grad().unwrap();
    let grad_slice: &[f32] = grad_coeus2.as_slice();
    for &g in grad_slice {
        assert!(g == 0.0 || (g - 2.0f32).abs() < 1e-5);
    }
}

#[test]
fn test_batchnorm2d_parity() {
    let x_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 0.5, 1.5, 2.5, 3.5,
        4.5, 5.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5,
    ];
    let w_data = vec![1.2f32, 0.8];
    let b_data = vec![0.1f32, -0.1];
    let eps = 1e-5;
    let momentum = 0.1;

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 2, 2, 3], &x_data), true);
    let mut bn_coeus =
        coeus_nn::normalization::batchnorm2d::BatchNorm2d::<f32, SequentialBackend>::new(
            2, eps, momentum,
        );
    bn_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![2], &w_data), true);
    bn_coeus.bias = CoeusVar::new(CoeusTensor::from_slice(vec![2], &b_data), true);
    let out_coeus = bn_coeus.forward(&x_coeus);

    // Verify forward
    let expected_batchnorm2d_out = vec![
        -1.464284f32,
        -0.769047f32,
        -0.073809f32,
        0.621428f32,
        1.316666f32,
        2.011903f32,
        0.942856f32,
        0.479364f32,
        0.015873f32,
        -0.447619f32,
        -0.911110f32,
        -1.374602f32,
        -1.811903f32,
        -1.116665f32,
        -0.421428f32,
        0.273809f32,
        0.969047f32,
        1.664284f32,
        1.174602f32,
        0.711110f32,
        0.247619f32,
        -0.215873f32,
        -0.679365f32,
        -1.142856f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_batchnorm2d_out, 1e-3);

    // Backward
    out_coeus.backward();

    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = bn_coeus.weight.grad().unwrap();
    let db_coeus = bn_coeus.bias.grad().unwrap();

    let expected_batchnorm2d_dx = vec![0.000000f32; 24];
    let expected_batchnorm2d_dw = vec![0.000000f32, -0.000000f32];
    let expected_batchnorm2d_db = vec![12.000000f32, 12.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_batchnorm2d_dx, 1e-3);
    assert_tensor_eq_data(&dw_coeus, &expected_batchnorm2d_dw, 1e-3);
    assert_tensor_eq_data(&db_coeus, &expected_batchnorm2d_db, 1e-3);
}

#[test]
fn test_conv2d_parity() {
    // Conv2D: batch=1, in_channels=2, height=3, width=3, out_channels=2, kernel=2
    let x_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
        -8.0, -9.0,
    ];
    let w_data = vec![
        0.5f32, -0.5, 1.0, 0.0, 0.1f32, 0.2, 0.3, -0.1, -0.2f32, 0.5, 0.0, 1.0, 1.0f32, -1.0, 0.2,
        0.8,
    ];
    let b_data = vec![0.1f32, -0.2];

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![1, 2, 3, 3], &x_data), true);
    let mut conv_coeus = CoeusConv2d::<f32, SequentialBackend>::with_params(2, 2, 2, 1, 0, 1, true);
    conv_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![2, 2, 2, 2], &w_data), true);
    conv_coeus.bias = Some(CoeusVar::new(
        CoeusTensor::from_slice(vec![2], &b_data),
        true,
    ));

    // Coeus forward
    let out_coeus = conv_coeus.forward(&x_coeus);

    // Verify forward
    let expected_conv2d_out = vec![
        2.400000f32,
        2.900000f32,
        3.900000f32,
        4.400000f32,
        1.800000f32,
        2.100000f32,
        2.700000f32,
        2.999999f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_conv2d_out, 1e-4);

    // Coeus backward
    out_coeus.backward();

    // Verify gradients
    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = conv_coeus.weight.grad().unwrap();
    let db_coeus = conv_coeus.bias.as_ref().unwrap().grad().unwrap();

    let expected_conv2d_dx = vec![
        0.300000f32,
        0.300000f32,
        0.000000f32,
        1.300000f32,
        2.300000f32,
        1.000000f32,
        1.000000f32,
        2.000000f32,
        1.000000f32,
        1.100000f32,
        0.300000f32,
        -0.800000f32,
        1.600000f32,
        1.500000f32,
        -0.100000f32,
        0.500000f32,
        1.200000f32,
        0.700000f32,
    ];
    let expected_conv2d_dw = vec![
        12.000000f32,
        16.000000f32,
        24.000000f32,
        28.000000f32,
        -12.000000f32,
        -16.000000f32,
        -24.000000f32,
        -28.000000f32,
        12.000000f32,
        16.000000f32,
        24.000000f32,
        28.000000f32,
        -12.000000f32,
        -16.000000f32,
        -24.000000f32,
        -28.000000f32,
    ];
    let expected_conv2d_db = vec![4.000000f32, 4.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_conv2d_dx, 1e-4);
    assert_tensor_eq_data(&dw_coeus, &expected_conv2d_dw, 1e-4);
    assert_tensor_eq_data(&db_coeus, &expected_conv2d_db, 1e-4);
}

#[test]
fn test_conv3d_parity() {
    // Conv3D: batch=1, in_channels=1, depth=2, height=2, width=2, out_channels=1, kernel=2
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w_data = vec![0.5f32, -0.5, 1.0, 0.0, 0.1, 0.2, 0.3, -0.1];
    let b_data = vec![0.1f32];

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![1, 1, 2, 2, 2], &x_data), true);
    let mut conv_coeus = CoeusConv3d::<f32, SequentialBackend>::with_params(1, 1, 2, 1, 0, 1, true);
    conv_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![1, 1, 2, 2, 2], &w_data), true);
    conv_coeus.bias = Some(CoeusVar::new(
        CoeusTensor::from_slice(vec![1], &b_data),
        true,
    ));

    // Coeus forward
    let out_coeus = conv_coeus.forward(&x_coeus);

    // Verify forward
    let expected_conv3d_out = vec![5.600000f32];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_conv3d_out, 1e-4);

    // Coeus backward
    out_coeus.backward();

    // Verify gradients
    let dx_coeus = x_coeus.grad().unwrap();
    let dw_coeus = conv_coeus.weight.grad().unwrap();
    let db_coeus = conv_coeus.bias.as_ref().unwrap().grad().unwrap();

    let expected_conv3d_dx = vec![
        0.500000f32,
        -0.500000f32,
        1.000000f32,
        0.000000f32,
        0.100000f32,
        0.200000f32,
        0.300000f32,
        -0.100000f32,
    ];
    let expected_conv3d_dw = vec![
        1.000000f32,
        2.000000f32,
        3.000000f32,
        4.000000f32,
        5.000000f32,
        6.000000f32,
        7.000000f32,
        8.000000f32,
    ];
    let expected_conv3d_db = vec![1.000000f32];

    assert_tensor_eq_data(&dx_coeus, &expected_conv3d_dx, 1e-4);
    assert_tensor_eq_data(&dw_coeus, &expected_conv3d_dw, 1e-4);
    assert_tensor_eq_data(&db_coeus, &expected_conv3d_db, 1e-4);
}

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
    loss_coeus.backward();

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
    let loss_coeus = coeus_nn::cross_entropy_loss(&logits_coeus, &targets_data);

    // Verify forward (Mean Cross Entropy: mean(-log(softmax(logits)[target])))
    let expected_cross_entropy_out = vec![0.288726f32];
    assert_tensor_eq_data::<SequentialBackend>(
        &loss_coeus.tensor,
        &expected_cross_entropy_out,
        1e-4,
    );

    // Backward
    loss_coeus.backward();

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
fn test_mha_parity() {
    let backend = SequentialBackend::new();

    // Query, key, value shape: [2, 3, 8] (batch=2, seq=3, d_model=8)
    // Heads = 2, so H = 2.

    let q_data = vec![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7,
    ];
    let k_data = q_data.clone();
    let v_data = q_data.clone();

    let wq_data = vec![0.1f32; 64];
    let bq_data = vec![0.05f32; 8];
    let wk_data = vec![0.2f32; 64];
    let bk_data = vec![0.1f32; 8];
    let wv_data = vec![0.3f32; 64];
    let bv_data = vec![0.15f32; 8];
    let wo_data = vec![0.4f32; 64];
    let bo_data = vec![0.2f32; 8];

    // Coeus setup
    let q_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &q_data),
        true,
    );
    let k_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &k_data),
        true,
    );
    let v_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &v_data),
        true,
    );

    let mut mha_coeus = coeus_nn::MultiHeadAttention::<f32, SequentialBackend, 2>::new(8, true);
    mha_coeus.w_q = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wq_data),
        true,
    );
    mha_coeus.b_q = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bq_data),
        true,
    ));
    mha_coeus.w_k = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wk_data),
        true,
    );
    mha_coeus.b_k = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bk_data),
        true,
    ));
    mha_coeus.w_v = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wv_data),
        true,
    );
    mha_coeus.b_v = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bv_data),
        true,
    ));
    mha_coeus.w_o = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wo_data),
        true,
    );
    mha_coeus.b_o = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bo_data),
        true,
    ));

    let out_coeus = mha_coeus.forward_cross(&q_coeus, &k_coeus, &v_coeus, None);

    // Verify Q, K, V projections match
    let q_proj_coeus = {
        let flat = coeus_autograd::reshape(&q_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_q);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_q.as_ref().unwrap())
    };
    let k_proj_coeus = {
        let flat = coeus_autograd::reshape(&k_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_k);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_k.as_ref().unwrap())
    };
    let v_proj_coeus = {
        let flat = coeus_autograd::reshape(&v_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_v);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_v.as_ref().unwrap())
    };

    let q_split = coeus_autograd::reshape(&q_proj_coeus, [2, 3, 2, 4]);
    let q_perm = coeus_autograd::permute(&q_split, &[0, 2, 1, 3]);
    let q_heads = coeus_autograd::reshape(&q_perm, [4, 3, 4]);

    let k_split = coeus_autograd::reshape(&k_proj_coeus, [2, 3, 2, 4]);
    let k_perm = coeus_autograd::permute(&k_split, &[0, 2, 1, 3]);
    let k_heads = coeus_autograd::reshape(&k_perm, [4, 3, 4]);

    let v_split = coeus_autograd::reshape(&v_proj_coeus, [2, 3, 2, 4]);
    let v_perm = coeus_autograd::permute(&v_split, &[0, 2, 1, 3]);
    let v_heads = coeus_autograd::reshape(&v_perm, [4, 3, 4]);

    let (out_tensor, _attn_weights_coeus) = coeus_ops::scaled_dot_product_attention(
        &q_heads.tensor,
        &k_heads.tensor,
        &v_heads.tensor,
        None,
        false,
        0.5f32,
        &backend,
    );

    let out_var = CoeusVar::new(out_tensor, false);
    let merged_split = coeus_autograd::reshape(&out_var, [2, 2, 3, 4]);
    let merged_perm = coeus_autograd::permute(&merged_split, &[0, 2, 1, 3]);
    let _merged = coeus_autograd::reshape(&merged_perm, [2, 3, 8]);

    // Verify forward output
    let expected_mha_out = vec![
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_mha_out, 1e-4);

    // Backward
    let loss_coeus = coeus_autograd::sum(&out_coeus);
    loss_coeus.backward();

    // Verify input gradients
    let dq_coeus = q_coeus.grad().unwrap();
    let dk_coeus = k_coeus.grad().unwrap();
    let dv_coeus = v_coeus.grad().unwrap();

    let expected_mha_dq = vec![
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
    ];
    let expected_mha_dk = vec![
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
    ];
    let expected_mha_dv = vec![
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
    ];

    assert_tensor_eq_data(&dq_coeus, &expected_mha_dq, 1e-4);
    assert_tensor_eq_data(&dk_coeus, &expected_mha_dk, 1e-4);
    assert_tensor_eq_data(&dv_coeus, &expected_mha_dv, 1e-4);

    // Verify parameter gradients (note: PyTorch weight matrices are transposed compared to Coeus)
    let dwq_coeus = mha_coeus.w_q.grad().unwrap();
    let dbq_coeus = mha_coeus.b_q.as_ref().unwrap().grad().unwrap();
    let dwk_coeus = mha_coeus.w_k.grad().unwrap();
    let dbk_coeus = mha_coeus.b_k.as_ref().unwrap().grad().unwrap();
    let dwv_coeus = mha_coeus.w_v.grad().unwrap();
    let dbv_coeus = mha_coeus.b_v.as_ref().unwrap().grad().unwrap();
    let dwo_coeus = mha_coeus.w_o.grad().unwrap();
    let dbo_coeus = mha_coeus.b_o.as_ref().unwrap().grad().unwrap();

    let expected_mha_dwq = vec![
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350862f32,
        -3.350862f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
    ];
    let expected_mha_dbq = vec![
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
    ];
    let expected_mha_dwk = vec![
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807602f32,
        -0.807602f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
    ];
    let expected_mha_dbk = vec![
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
    ];
    let expected_mha_dwv = vec![
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341515f32,
        0.341515f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691730f32,
        0.691730f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041947f32,
        1.041947f32,
    ];
    let expected_mha_dbv = vec![
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
    ];
    let expected_mha_dwo = vec![
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
    ];
    let expected_mha_dbo = vec![6.0f32, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0];

    // Note: expected_mha_dwq, dwk, dwv, dwo in expected_values.txt are pre-transposed from PyTorch.
    // Let's transpose PyTorch expected to match Coeus's weight layout.
    // For MHA, the weights are stored as [8, 8] in both, but Coeus does flat @ w_q.T or w_q.
    // In our manual projection: w_t = transpose_2d(&mha_coeus.w_q) -> flat @ w_t
    // So the weight matrix in Coeus has shape [8, 8], where row-major is [in_features, out_features] or [out_features, in_features]
    // Let's verify by transposing the expected 8x8 matrix if needed.
    // The expected_mha_dwq from PyTorch was 8x8, where output was printed as row-major.
    // Let's check: PyTorch weight grad is shape [8, 8], and we printed it.
    // Coeus weight grad is compared to `dwq_burn.transpose()`, which means Coeus weight layout is the transpose of PyTorch's.
    // Therefore, in our self-contained test, we should transpose the expected PyTorch array to match Coeus weight grad!
    // PyTorch weight grad shape is [8, 8]. If we transpose it, the element at [row, col] goes to [col, row].
    // Let's write a simple helper to assert with transposition, or pre-transpose the vectors!
    // Let's write a transpose helper:
    let expected_mha_dwq_transposed = transpose_8x8(&expected_mha_dwq);
    let expected_mha_dwk_transposed = transpose_8x8(&expected_mha_dwk);
    let expected_mha_dwv_transposed = transpose_8x8(&expected_mha_dwv);
    let expected_mha_dwo_transposed = transpose_8x8(&expected_mha_dwo);

    assert_tensor_eq_data(&dwq_coeus, &expected_mha_dwq_transposed, 1e-4);
    assert_tensor_eq_data(&dbq_coeus, &expected_mha_dbq, 1e-4);
    assert_tensor_eq_data(&dwk_coeus, &expected_mha_dwk_transposed, 1e-4);
    assert_tensor_eq_data(&dbk_coeus, &expected_mha_dbk, 1e-4);
    assert_tensor_eq_data(&dwv_coeus, &expected_mha_dwv_transposed, 1e-4);
    assert_tensor_eq_data(&dbv_coeus, &expected_mha_dbv, 1e-4);
    assert_tensor_eq_data(&dwo_coeus, &expected_mha_dwo_transposed, 1e-4);
    assert_tensor_eq_data(&dbo_coeus, &expected_mha_dbo, 1e-4);
}

fn transpose_8x8(src: &[f32]) -> Vec<f32> {
    assert_eq!(src.len(), 64);
    let mut dst = vec![0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            dst[c * 8 + r] = src[r * 8 + c];
        }
    }
    dst
}
