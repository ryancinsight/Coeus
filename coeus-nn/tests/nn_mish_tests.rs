use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{mish, Mish, Module};
use coeus_tensor::{Tensor, Transpose};

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
    for (i, &x) in input_data.iter().enumerate() {
        let sp = (1.0 + x.exp()).ln();
        let expected = x * sp.tanh();
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward();
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // Gradient parity checks: tanh(softplus(x)) + x * (1 - tanh^2(softplus(x))) * sigmoid(x)
    for (i, &x) in input_data.iter().enumerate() {
        let sp = (1.0 + x.exp()).ln();
        let w = sp.tanh();
        let sig = 1.0 / (1.0 + (-x).exp());
        let expected_grad = w + x * (1.0 - w * w) * sig;
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }
}

#[test]
fn test_mish_module_cpu() {
    let mish_mod = Mish;
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0f64, 0.0, 1.0, 2.0]),
        true,
    );

    let output = mish_mod.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 2]);
    assert_eq!(Module::<f64, MoiraiBackend>::parameters(&mish_mod).len(), 0);

    output.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_mish_non_contiguous_cpu() {
    let input_raw =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let input_t = input_raw.transpose(); // shape [3, 2], non-contiguous
    let input = Var::new(input_t, true);

    let output = mish(&input);
    assert_eq!(output.tensor.shape(), &[3, 2]);

    output.backward();
    assert!(input.grad().is_some());
}
