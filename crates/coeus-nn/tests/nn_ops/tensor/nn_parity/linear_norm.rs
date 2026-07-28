use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_nn::{LayerNorm as CoeusLayerNorm, Linear as CoeusLinear, Module as CoeusModule};
use coeus_tensor::Tensor as CoeusTensor;

#[test]
fn test_linear_parity() {
    // Inputs: batch_size=2, in_features=3, out_features=2
    let x_data = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 2.5];
    let w_data = vec![0.5f32, -0.5, 1.0, 0.0, 2.0, -1.0];
    let b_data = vec![0.2f32, -0.1];

    // Coeus setup
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 3], &x_data).expect("construct tensor"), true).expect("construct variable");
    let mut linear_coeus = CoeusLinear::<f32, SequentialBackend>::new(3, 2, true).expect("construct module");
    linear_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![2, 3], &w_data).expect("construct tensor"), true).expect("construct variable");
    linear_coeus.bias = Some(CoeusVar::new(
        CoeusTensor::from_slice(vec![2], &b_data).expect("construct tensor"),
        true,
    ).expect("construct variable"));

    // Coeus forward
    let out_coeus = linear_coeus.forward(&x_coeus).expect("run forward");

    // Verify forward
    let expected_linear_out = vec![2.700000f32, 0.900000f32, 1.950000f32, -1.600000f32];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_linear_out, 1e-4);

    // Coeus backward
    out_coeus.backward().expect("run backward");

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
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 4], &x_data).expect("construct tensor"), true).expect("construct variable");
    let mut ln_coeus = CoeusLayerNorm::<f32, SequentialBackend>::new(4, eps).expect("construct module");
    ln_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![4], &w_data).expect("construct tensor"), true).expect("construct variable");
    ln_coeus.bias = CoeusVar::new(CoeusTensor::from_slice(vec![4], &b_data).expect("construct tensor"), true).expect("construct variable");

    // Coeus forward
    let out_coeus = ln_coeus.forward(&x_coeus).expect("run forward");

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
    out_coeus.backward().expect("run backward");

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
