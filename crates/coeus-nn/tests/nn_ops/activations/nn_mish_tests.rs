use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{mish, Mish, Module};
use coeus_tensor::{Tensor, Transpose};

fn mish_expected(x: f64) -> f64 {
    let sp = (1.0 + x.exp()).ln();
    x * sp.tanh()
}

fn mish_grad_expected(x: f64) -> f64 {
    let sp = (1.0 + x.exp()).ln();
    let w = sp.tanh();
    let sig = 1.0 / (1.0 + (-x).exp());
    w + x * (1.0 - w * w) * sig
}

fn assert_close(label: &str, got: f64, expected: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff < 1e-7,
        "{label}: got={got:.12}, expected={expected:.12}, diff={diff:.3e}"
    );
}

fn assert_mish_values(label: &str, got: &[f64], input: &[f64]) {
    assert_eq!(got.len(), input.len(), "{label}: length mismatch");
    for (i, (&actual, &x)) in got.iter().zip(input).enumerate() {
        assert_close(&format!("{label}[{i}]"), actual, mish_expected(x));
    }
}

fn assert_mish_grads(label: &str, got: &[f64], input: &[f64]) {
    assert_eq!(got.len(), input.len(), "{label}: length mismatch");
    for (i, (&actual, &x)) in got.iter().zip(input).enumerate() {
        assert_close(&format!("{label}[{i}]"), actual, mish_grad_expected(x));
    }
}

#[test]
fn test_mish_functional_cpu() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data),
        true,
    );

    let output = mish(&input);
    assert_eq!(output.tensor.shape(), &[5]);

    // Value parity checks: x * tanh(softplus(x))
    let out_slice = output.tensor.as_slice();
    assert_mish_values("functional_forward", out_slice, &input_data);

    // Backward pass
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // Gradient parity checks: tanh(softplus(x)) + x * (1 - tanh^2(softplus(x))) * sigmoid(x)
    assert_mish_grads("functional_backward", &grad_slice, &input_data);
}

#[test]
fn test_mish_module_cpu() {
    let mish_mod = Mish;
    let input_data = [-1.0f64, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &input_data),
        true,
    );

    let output = mish_mod.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 2]);
    assert_mish_values("module_forward", output.tensor.as_slice(), &input_data);
    assert_eq!(Module::<f64, MoiraiBackend>::parameters(&mish_mod).len(), 0);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().expect("invariant: Mish input requires grad");
    assert_mish_grads("module_backward", grad.as_slice(), &input_data);
}

#[test]
fn test_mish_non_contiguous_cpu() {
    let input_raw =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let input_t = input_raw.transpose(); // shape [3, 2], non-contiguous
    let logical_input = [-2.0f64, 1.0, -1.0, 2.0, 0.0, 3.0];
    let input = Var::new(input_t, true);

    let output = mish(&input);
    assert_eq!(output.tensor.shape(), &[3, 2]);
    assert_mish_values(
        "non_contiguous_forward",
        output.tensor.as_slice(),
        &logical_input,
    );

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input
        .grad()
        .expect("invariant: non-contiguous Mish input requires grad");
    assert_mish_grads("non_contiguous_backward", grad.as_slice(), &logical_input);
}
