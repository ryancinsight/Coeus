use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_nn::{
    Conv1d as CoeusConv1d, Conv2d as CoeusConv2d, Conv3d as CoeusConv3d, Module as CoeusModule,
};
use coeus_tensor::Tensor as CoeusTensor;

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
    out_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

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
    out_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

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
    out_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

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
