use super::{Adam, AdamW, Optimizer, Parameter, RMSProp, SequentialBackend, Tensor, Var, SGD};

fn failure_atomic_parameters() -> Vec<Parameter<f32, SequentialBackend>> {
    let first = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0]),
        true,
    );
    first.set_grad(Tensor::from_slice(vec![1], &[1.0]));
    let second = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[3.0]),
        true,
    );
    second.set_grad(Tensor::from_slice(vec![2], &[1.0, 1.0]));
    vec![
        Parameter::new(first, "first"),
        Parameter::new(second, "second"),
    ]
}

fn assert_failed_pair_unchanged(params: &[Parameter<f32, SequentialBackend>]) {
    assert_eq!(params[0].tensor.as_slice(), &[2.0]);
    assert_eq!(params[1].tensor.as_slice(), &[3.0]);
}

fn repair_second_gradient(params: &mut [Parameter<f32, SequentialBackend>]) {
    params[1].set_grad(Tensor::from_slice(vec![1], &[1.0]));
}

#[test]
fn test_sgd_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test SGD step without momentum (momentum = 0.0, lr = 0.1)
    let mut optimizer = SGD::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 0.0f32);
    optimizer.step().expect("SGD step");
    assert_eq!(optimizer.params[0].name, "x");

    // After one step, param = param - lr * grad
    // x[0] = 2.0 - 0.1 * 1.0 = 1.9
    // x[1] = 3.0 - 0.1 * (-2.0) = 3.2
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-5);
    assert!((updated_x[1] - 3.2).abs() < 1e-5);

    // Verify zero_grad works
    optimizer.zero_grad();
    let cleared_grad = optimizer.params[0].grad().unwrap();
    assert_eq!(cleared_grad.as_slice(), &[0.0, 0.0]);
}

#[test]
fn test_sgd_with_momentum() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Let's perform two steps of SGD with momentum = 0.9, lr = 0.1
    let mut optimizer = SGD::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 0.9f32);

    // Step 1
    // grad = [1.0, -2.0]
    // v = momentum * 0 + grad = [1.0, -2.0]
    // param = param - lr * v = [2.0, 3.0] - 0.1 * [1.0, -2.0] = [1.9, 3.2]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    optimizer.params[0].set_grad(grad_val);
    optimizer.step().expect("SGD momentum step");
    assert_eq!(optimizer.params[0].name, "x");

    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-5);
    assert!((updated_x[1] - 3.2).abs() < 1e-5);

    // Step 2
    // grad = [0.5, 0.5]
    // v_prev = [1.0, -2.0]
    // v_new = 0.9 * v_prev + grad = 0.9 * [1.0, -2.0] + [0.5, 0.5] = [1.4, -1.3]
    // param = param - lr * v_new = [1.9, 3.2] - 0.1 * [1.4, -1.3] = [1.76, 3.33]
    let grad_val2 = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[0.5f32, 0.5]);
    optimizer.params[0].set_grad(grad_val2);
    optimizer.step().expect("second SGD momentum step");
    assert_eq!(optimizer.params[0].name, "x");

    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.76).abs() < 1e-5);
    assert!((updated_x[1] - 3.33).abs() < 1e-5);
}

#[test]
fn test_adam_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test Adam (lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    let mut optimizer = Adam::new(
        vec![Parameter::new(x.clone(), "x")],
        0.1f32,
        0.9f32,
        0.999f32,
        1e-8f32,
    );
    optimizer.step().expect("Adam step");
    assert_eq!(optimizer.params[0].name, "x");

    // After step 1:
    // t = 1
    // beta1^t = 0.9, beta2^t = 0.999
    // m = beta1 * 0 + (1 - beta1) * grad = 0.1 * [1.0, -2.0] = [0.1, -0.2]
    // v = beta2 * 0 + (1 - beta2) * grad^2 = 0.001 * [1.0, 4.0] = [0.001, 0.004]
    // m_hat = m / (1 - beta1) = [0.1, -0.2] / 0.1 = [1.0, -2.0]
    // v_hat = v / (1 - beta2) = [0.001, 0.004] / 0.001 = [1.0, 4.0]
    // update = lr * m_hat / (sqrt(v_hat) + eps)
    //        = 0.1 * [1.0, -2.0] / ([1.0, 2.0] + 1e-8)
    //        ≈ 0.1 * [1.0, -1.0] = [0.1, -0.1]
    // param = param - update = [2.0, 3.0] - [0.1, -0.1] = [1.9, 3.1]
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-4);
    assert!((updated_x[1] - 3.1).abs() < 1e-4);
}

#[test]
fn test_rmsprop_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test RMSProp (lr = 0.1, alpha = 0.99, eps = 1e-8)
    let mut optimizer = RMSProp::new(
        vec![Parameter::new(x.clone(), "x")],
        0.1f32,
        0.99f32,
        1e-8f32,
    );
    optimizer.step().expect("RMSProp step");
    assert_eq!(optimizer.params[0].name, "x");

    // After step 1:
    // v = alpha * 0 + (1 - alpha) * grad^2 = 0.01 * [1.0, 4.0] = [0.01, 0.04]
    // denom = sqrt(v) + eps = [0.1, 0.2] + 1e-8
    // update = lr * grad / denom = 0.1 * [1.0, -2.0] / [0.1, 0.2] = [1.0, -1.0]
    // param = param - update = [2.0, 3.0] - [1.0, -1.0] = [1.0, 4.0]
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.0).abs() < 1e-4);
    assert!((updated_x[1] - 4.0).abs() < 1e-4);
}

#[test]
fn test_adamw_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test AdamW (lr = 0.1, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    let mut optimizer = AdamW::new(
        vec![Parameter::new(x.clone(), "x")],
        0.1f32,
        0.9f32,
        0.999f32,
        1e-8f32,
        0.01f32,
    );
    optimizer.step().expect("AdamW step");

    // After step 1:
    // t = 1
    // beta1^t = 0.9, beta2^t = 0.999
    // m = beta1 * 0 + (1 - beta1) * grad = 0.1 * [1.0, -2.0] = [0.1, -0.2]
    // v = beta2 * 0 + (1 - beta2) * grad^2 = 0.001 * [1.0, 4.0] = [0.001, 0.004]
    // m_hat = m / (1 - beta1) = [0.1, -0.2] / 0.1 = [1.0, -2.0]
    // v_hat = v / (1 - beta2) = [0.001, 0.004] / 0.001 = [1.0, 4.0]
    // adam_update = lr * m_hat / (sqrt(v_hat) + eps) ≈ 0.1 * [1.0, -1.0] = [0.1, -0.1]
    // wd_update = lr * weight_decay * param = 0.1 * 0.01 * [2.0, 3.0] = [0.002, 0.003]
    // param = param - adam_update - wd_update
    // x[0] = 2.0 - 0.1 - 0.002 = 1.898
    // x[1] = 3.0 - (-0.1) - 0.003 = 3.097
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.898).abs() < 1e-4);
    assert!((updated_x[1] - 3.097).abs() < 1e-4);
}

#[test]
fn test_adagrad_optimizer() {
    let _backend = SequentialBackend::new();
    let x_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[2.0f32, 3.0]);
    let x = Var::new(x_val, true);

    // Set mock gradient: [1.0, -2.0]
    let grad_val = Tensor::<f32, SequentialBackend>::from_slice(vec![2], &[1.0f32, -2.0]);
    x.set_grad(grad_val);

    // Test AdaGrad (lr = 0.1, eps = 1e-6)
    let mut optimizer =
        coeus_optim::AdaGrad::new(vec![Parameter::new(x.clone(), "x")], 0.1f32, 1e-6f32);
    optimizer.step().expect("AdaGrad step");

    // After step 1:
    // history = history + grad^2 = [1.0, 4.0]
    // denom = sqrt(history) + eps = [1.000001, 2.000001]
    // update = lr * grad / denom = 0.1 * [1.0, -2.0] / [1.000001, 2.000001]
    // param = param - update
    let updated_x = optimizer.params[0].tensor.as_slice();
    assert!((updated_x[0] - 1.9).abs() < 1e-4);
    assert!((updated_x[1] - 3.1).abs() < 1e-4);
}

#[test]
fn failed_adam_family_steps_preserve_bias_counter() {
    let gradient = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[1.0]);

    let adam_var = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0]),
        true,
    );
    adam_var.set_grad(gradient.clone());
    let mut adam = Adam::new(vec![Parameter::new(adam_var, "adam")], 0.1, 0.9, 0.999, 0.0);
    adam.step()
        .expect_err("zero Adam epsilon must reject the update");
    assert_eq!(adam.t, 0);
    assert_eq!(adam.params[0].tensor.as_slice(), &[2.0]);
    assert_eq!(adam.m[0].as_slice(), &[0.0]);
    assert_eq!(adam.v[0].as_slice(), &[0.0]);
    adam.eps = 1.0e-8;
    adam.step().expect("retry Adam step");
    assert_eq!(adam.t, 1);
    assert!((adam.params[0].tensor.as_slice()[0] - 1.9).abs() < 1.0e-4);

    let adamw_var = Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(vec![1], &[2.0]),
        true,
    );
    adamw_var.set_grad(gradient);
    let mut adamw = AdamW::new(
        vec![Parameter::new(adamw_var, "adamw")],
        0.1,
        0.9,
        0.999,
        0.0,
        0.01,
    );
    adamw
        .step()
        .expect_err("zero AdamW epsilon must reject the update");
    assert_eq!(adamw.t, 0);
    assert_eq!(adamw.params[0].tensor.as_slice(), &[2.0]);
    assert_eq!(adamw.m[0].as_slice(), &[0.0]);
    assert_eq!(adamw.v[0].as_slice(), &[0.0]);
    adamw.eps = 1.0e-8;
    adamw.step().expect("retry AdamW step");
    assert_eq!(adamw.t, 1);
    assert!((adamw.params[0].tensor.as_slice()[0] - 1.898).abs() < 1.0e-4);
}

#[test]
fn multi_parameter_validation_is_failure_atomic() {
    let mut sgd = SGD::new(failure_atomic_parameters(), 0.1, 0.9);
    sgd.step()
        .expect_err("second SGD shape must fail preflight");
    assert_failed_pair_unchanged(&sgd.params);
    repair_second_gradient(&mut sgd.params);
    sgd.step().expect("repaired SGD step");
    assert!((sgd.params[0].tensor.as_slice()[0] - 1.9).abs() < 1.0e-6);

    let mut adam = Adam::new(failure_atomic_parameters(), 0.1, 0.9, 0.999, 1.0e-8);
    adam.step()
        .expect_err("second Adam shape must fail preflight");
    assert_failed_pair_unchanged(&adam.params);
    assert_eq!(adam.t, 0);
    repair_second_gradient(&mut adam.params);
    adam.step().expect("repaired Adam step");
    assert_eq!(adam.t, 1);
    assert!((adam.params[0].tensor.as_slice()[0] - 1.9).abs() < 1.0e-4);

    let mut rmsprop = RMSProp::new(failure_atomic_parameters(), 0.1, 0.99, 1.0e-8);
    rmsprop
        .step()
        .expect_err("second RMSProp shape must fail preflight");
    assert_failed_pair_unchanged(&rmsprop.params);
    repair_second_gradient(&mut rmsprop.params);
    rmsprop.step().expect("repaired RMSProp step");
    assert!((rmsprop.params[0].tensor.as_slice()[0] - 1.0).abs() < 1.0e-4);

    let mut adamw = AdamW::new(failure_atomic_parameters(), 0.1, 0.9, 0.999, 1.0e-8, 0.01);
    adamw
        .step()
        .expect_err("second AdamW shape must fail preflight");
    assert_failed_pair_unchanged(&adamw.params);
    assert_eq!(adamw.t, 0);
    repair_second_gradient(&mut adamw.params);
    adamw.step().expect("repaired AdamW step");
    assert_eq!(adamw.t, 1);
    assert!((adamw.params[0].tensor.as_slice()[0] - 1.898).abs() < 1.0e-4);

    let mut adagrad = coeus_optim::AdaGrad::new(failure_atomic_parameters(), 0.1, 1.0e-6);
    adagrad
        .step()
        .expect_err("second AdaGrad shape must fail preflight");
    assert_failed_pair_unchanged(&adagrad.params);
    repair_second_gradient(&mut adagrad.params);
    adagrad.step().expect("repaired AdaGrad step");
    assert!((adagrad.params[0].tensor.as_slice()[0] - 1.9).abs() < 1.0e-4);
}
