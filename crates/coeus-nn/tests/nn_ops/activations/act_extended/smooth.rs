use super::support::assert_close_slice;
use super::{log_sigmoid, tanhshrink, MoiraiBackend, Tensor, Var};

// ── LogSigmoid ───────────────────────────────────────────────────────────────

#[test]
fn log_sigmoid_forward_and_backward() {
    // logsigmoid(x) = log(sigmoid(x)) = -log(1 + e^-x);
    // d/dx = sigmoid(-x) = 1 / (1 + e^x).
    let data = vec![-4.0_f64, -1.0, 0.0, 1.0, 3.0];
    let expected: Vec<f64> = data.iter().map(|&x| -(1.0 + (-x).exp()).ln()).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| 1.0 / (1.0 + x.exp())).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = log_sigmoid(&input).expect("run operation");
    // Tolerance: stable identity vs coeus's `-softplus(-x)`; agreement is ~1e-15
    // for moderate x, 1e-10 is a safe margin.
    assert_close_slice(
        "log_sigmoid_forward",
        output.tensor.as_slice(),
        &expected,
        1e-10,
    );
    output.backward().expect("run backward");
    let grad = input.grad().expect("log_sigmoid requires grad");
    assert_close_slice(
        "log_sigmoid_backward",
        grad.as_slice(),
        &expected_grad,
        1e-10,
    );
}

// ── Tanhshrink ───────────────────────────────────────────────────────────────

#[test]
fn tanhshrink_forward_and_backward() {
    // tanhshrink(x) = x - tanh(x);  d/dx = 1 - sech^2(x) = tanh^2(x).
    let data = vec![-3.0_f64, -1.0, 0.0, 0.5, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| x - x.tanh()).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| x.tanh().powi(2)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = tanhshrink(&input).expect("run operation");
    assert_close_slice(
        "tanhshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
    let grad = input.grad().expect("tanhshrink requires grad");
    assert_close_slice(
        "tanhshrink_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}
