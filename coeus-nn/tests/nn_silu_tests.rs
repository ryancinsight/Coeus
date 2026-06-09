use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{silu, Module, SiLU};
use coeus_tensor::{Tensor, Transpose};

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
    for (i, &x) in input_data.iter().enumerate() {
        let sig = 1.0 / (1.0 + (-x).exp());
        let expected = x * sig;
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward();
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // Gradient parity checks: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    for (i, &x) in input_data.iter().enumerate() {
        let sig = 1.0 / (1.0 + (-x).exp());
        let expected_grad = sig * (1.0 + x * (1.0 - sig));
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }
}

#[test]
fn test_silu_module_cpu() {
    let silu_mod = SiLU;
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0f64, 0.0, 1.0, 2.0]),
        true,
    );

    let output = silu_mod.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 2]);
    assert_eq!(Module::<f64, MoiraiBackend>::parameters(&silu_mod).len(), 0);

    output.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_silu_non_contiguous_cpu() {
    let input_raw =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let input_t = input_raw.transpose(); // shape [3, 2], non-contiguous
    let input = Var::new(input_t, true);

    let output = silu(&input);
    assert_eq!(output.tensor.shape(), &[3, 2]);

    output.backward();
    assert!(input.grad().is_some());
}
