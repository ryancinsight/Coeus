use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{silu, Module, SiLU};
use coeus_tensor::{Tensor, Transpose};

fn silu_expected(x: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-x).exp());
    x * sig
}

fn silu_grad_expected(x: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

fn assert_close(label: &str, got: f64, expected: f64) {
    let diff = (got - expected).abs();
    assert!(
        diff < 1e-7,
        "{label}: got={got:.12}, expected={expected:.12}, diff={diff:.3e}"
    );
}

fn assert_silu_values(label: &str, got: &[f64], input: &[f64]) {
    assert_eq!(got.len(), input.len(), "{label}: length mismatch");
    for (i, (&actual, &x)) in got.iter().zip(input).enumerate() {
        assert_close(&format!("{label}[{i}]"), actual, silu_expected(x));
    }
}

fn assert_silu_grads(label: &str, got: &[f64], input: &[f64]) {
    assert_eq!(got.len(), input.len(), "{label}: length mismatch");
    for (i, (&actual, &x)) in got.iter().zip(input).enumerate() {
        assert_close(&format!("{label}[{i}]"), actual, silu_grad_expected(x));
    }
}

#[test]
fn test_silu_functional_cpu() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data),
        true,
    );

    let output = silu(&input);
    assert_eq!(output.tensor.shape(), &[5]);

    // Value parity checks: x * sigmoid(x)
    let out_slice = output.tensor.as_slice();
    assert_silu_values("functional_forward", out_slice, &input_data);

    // Backward pass
    output.backward();
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // Gradient parity checks: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    assert_silu_grads("functional_backward", &grad_slice, &input_data);
}

#[test]
fn test_silu_module_cpu() {
    let silu_mod = SiLU;
    let input_data = [-1.0f64, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &input_data),
        true,
    );

    let output = silu_mod.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 2]);
    assert_silu_values("module_forward", output.tensor.as_slice(), &input_data);
    assert_eq!(Module::<f64, MoiraiBackend>::parameters(&silu_mod).len(), 0);

    output.backward();
    let grad = input.grad().expect("invariant: SiLU input requires grad");
    assert_silu_grads("module_backward", grad.as_slice(), &input_data);
}

#[test]
fn test_silu_non_contiguous_cpu() {
    let input_raw =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let input_t = input_raw.transpose(); // shape [3, 2], non-contiguous
    let logical_input = [-2.0f64, 1.0, -1.0, 2.0, 0.0, 3.0];
    let input = Var::new(input_t, true);

    let output = silu(&input);
    assert_eq!(output.tensor.shape(), &[3, 2]);
    assert_silu_values(
        "non_contiguous_forward",
        output.tensor.as_slice(),
        &logical_input,
    );

    output.backward();
    let grad = input
        .grad()
        .expect("invariant: non-contiguous SiLU input requires grad");
    assert_silu_grads("non_contiguous_backward", grad.as_slice(), &logical_input);
}
