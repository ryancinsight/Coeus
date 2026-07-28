use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_nn::Module as CoeusModule;
use coeus_tensor::Tensor as CoeusTensor;

#[test]
fn test_dropout_parity() {
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let p = 0.0;

    // Coeus setup
    let x_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &x_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut dropout_coeus = coeus_nn::Dropout::new(p);
    dropout_coeus.is_training = true;
    let out_coeus = dropout_coeus.forward(&x_coeus).expect("run forward");

    // Verify forward (with p=0, output is equal to input)
    assert_tensor_eq_data(&out_coeus.tensor, &x_data, 1e-4);

    // Backward
    out_coeus.backward().expect("run backward");

    let dx_coeus = x_coeus.grad().unwrap();

    // Verify backward (with p=0, output gradient is 1.0 for each element, so sum loss gradient is 1.0 for each)
    let expected_dx = vec![1.0f32; 6];
    assert_tensor_eq_data(&dx_coeus, &expected_dx, 1e-4);

    // Stochastic scaling test in Coeus
    let x_coeus2 = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![1000], &vec![1.0f32; 1000]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let mut dropout_coeus2 = coeus_nn::Dropout::new(0.5);
    dropout_coeus2.is_training = true;
    let out_coeus2 = dropout_coeus2.forward(&x_coeus2).expect("run forward");
    out_coeus2.backward().expect("run backward");

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
    let x_coeus = CoeusVar::new(CoeusTensor::from_slice(vec![2, 2, 2, 3], &x_data).expect("construct tensor"), true).expect("construct variable");
    let mut bn_coeus =
        coeus_nn::normalization::batchnorm2d::BatchNorm2d::<f32, SequentialBackend>::new(
            2, eps, momentum,
        ).expect("construct module");
    bn_coeus.weight = CoeusVar::new(CoeusTensor::from_slice(vec![2], &w_data).expect("construct tensor"), true).expect("construct variable");
    bn_coeus.bias = CoeusVar::new(CoeusTensor::from_slice(vec![2], &b_data).expect("construct tensor"), true).expect("construct variable");
    let out_coeus = bn_coeus.forward(&x_coeus).expect("run forward");

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
    out_coeus.backward().expect("run backward");

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
