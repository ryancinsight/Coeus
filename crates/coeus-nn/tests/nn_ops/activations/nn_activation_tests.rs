use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{
    elu, gelu_tanh, glu, leaky_relu, softplus, GeLUTanh, LeakyReLU, Module, Softplus, ELU, GLU,
};
use coeus_tensor::{Tensor, Transpose};

#[test]
fn test_elu_activation() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    // Test Functional
    let output = elu(&input).expect("run operation");
    assert_eq!(output.tensor.shape(), &[5]);
    let out_slice = output.tensor.as_slice();

    // ELU forward: x >= 0 ? x : exp(x) - 1
    for (i, &x) in input_data.iter().enumerate() {
        let expected = if x >= 0.0 { x } else { x.exp() - 1.0 };
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // ELU derivative: x >= 0 ? 1 : exp(x)
    for (i, &x) in input_data.iter().enumerate() {
        let expected_grad = if x >= 0.0 { 1.0 } else { x.exp() };
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }

    // Test Module
    let input_mod = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, 0.0, 1.0, 2.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let elu_mod = ELU;
    let output_mod = elu_mod.forward(&input_mod).expect("run forward");
    assert_eq!(output_mod.tensor.shape(), &[2, 2]);
    assert_eq!(Module::<f64, MoiraiBackend>::parameters(&elu_mod).len(), 0);
}

#[test]
fn test_softplus_activation() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    // Test Functional
    let output = softplus(&input).expect("run operation");
    assert_eq!(output.tensor.shape(), &[5]);
    let out_slice = output.tensor.as_slice();

    // Softplus forward: log(1 + exp(x))
    for (i, &x) in input_data.iter().enumerate() {
        let expected = (1.0 + x.exp()).ln();
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // Softplus derivative: sigmoid(x) = 1 / (1 + exp(-x))
    for (i, &x) in input_data.iter().enumerate() {
        let expected_grad = 1.0 / (1.0 + (-x).exp());
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }

    // Test Module
    let input_mod = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, 0.0, 1.0, 2.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let softplus_mod = Softplus;
    let output_mod = softplus_mod.forward(&input_mod).expect("run forward");
    assert_eq!(output_mod.tensor.shape(), &[2, 2]);
    assert_eq!(
        Module::<f64, MoiraiBackend>::parameters(&softplus_mod).len(),
        0
    );
}

#[test]
fn test_gelu_tanh_activation() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    // Test Functional
    let output = gelu_tanh(&input).expect("run operation");
    assert_eq!(output.tensor.shape(), &[5]);
    let out_slice = output.tensor.as_slice();

    // GELU Tanh forward: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c1 = 0.7978845608; // sqrt(2/pi)
    let c2 = 0.044715;
    for (i, &x) in input_data.iter().enumerate() {
        let v = c1 * (x + c2 * x * x * x);
        let expected = 0.5 * x * (1.0 + v.tanh());
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // GELU Tanh derivative:
    let c3 = 3.0 * c2;
    for (i, &x) in input_data.iter().enumerate() {
        let v = c1 * (x + c2 * x * x * x);
        let t = v.tanh();
        let dt = c1 * (1.0 + c3 * x * x);
        let expected_grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dt;
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }

    // Test Module
    let input_mod = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, 0.0, 1.0, 2.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let gelu_tanh_mod = GeLUTanh;
    let output_mod = gelu_tanh_mod.forward(&input_mod).expect("run forward");
    assert_eq!(output_mod.tensor.shape(), &[2, 2]);
    assert_eq!(
        Module::<f64, MoiraiBackend>::parameters(&gelu_tanh_mod).len(),
        0
    );
}

#[test]
fn test_leaky_relu_activation() {
    let input_data = vec![-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([5], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let slope = 0.1;

    // Test Functional
    let output = leaky_relu(&input, slope).expect("run operation");
    assert_eq!(output.tensor.shape(), &[5]);
    let out_slice = output.tensor.as_slice();

    // LeakyReLU forward: x >= 0 ? x : slope * x
    for (i, &x) in input_data.iter().enumerate() {
        let expected = if x >= 0.0 { x } else { slope * x };
        assert!((out_slice[i] - expected).abs() < 1e-7);
    }

    // Backward pass
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    let grad_slice = input.grad().unwrap().as_slice().to_vec();

    // LeakyReLU derivative: x > 0 ? 1 : slope
    // (Coeus contract for both LeakyReLU and PReLU matches PyTorch: slope at
    // the kink position x = 0, not 1. The forward predicate `x >= 0` still
    // gives 0 at x = 0; only the gradient predicate is tightened to mirror
    // PyTorch's `F.leaky_relu(neg_slope)` / `F.prelu` reduce-at-zero regime.)
    for (i, &x) in input_data.iter().enumerate() {
        let expected_grad = if x > 0.0 { 1.0 } else { slope };
        assert!((grad_slice[i] - expected_grad).abs() < 1e-7);
    }

    // Test Module
    let input_mod = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[-1.0, 0.0, 1.0, 2.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let leaky_mod = LeakyReLU::new(0.05);
    let output_mod = leaky_mod.forward(&input_mod).expect("run forward");
    assert_eq!(output_mod.tensor.shape(), &[2, 2]);
    assert_eq!(
        Module::<f64, MoiraiBackend>::parameters(&leaky_mod).len(),
        0
    );
}

#[test]
fn test_non_contiguous_activations() {
    let input_raw =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[-2.0f64, -1.0, 0.0, 1.0, 2.0, 3.0]).expect("construct tensor");
    let input_t = input_raw.transpose(); // shape [3, 2], non-contiguous
    let input = Var::new(input_t, true).expect("construct variable");

    // Test ELU on transposed view
    let output_elu = elu(&input).expect("run operation");
    assert_eq!(output_elu.tensor.shape(), &[3, 2]);
    output_elu.backward().expect("run backward");
    assert!(input.grad().is_some());

    // Reset gradient for next test
    let input2 = Var::new(input_raw.transpose(), true).expect("construct variable");
    let output_leaky = leaky_relu(&input2, 0.2).expect("run operation");
    assert_eq!(output_leaky.tensor.shape(), &[3, 2]);
    output_leaky.backward().expect("run backward");
    assert!(input2.grad().is_some());
}

#[test]
fn test_glu_forward_and_gradient() {
    // input=[1,2,3,4], dim=0 -> halves a=[1,2], b=[3,4]; glu = a * sigmoid(b).
    // out = [1*sigma(3), 2*sigma(4)].
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (s3, s4) = (sig(3.0), sig(4.0));

    let out = glu(&input, 0).expect("run operation");
    assert_eq!(out.tensor.shape(), &[2]);
    let o = out.tensor.as_slice();
    assert!((o[0] - 1.0 * s3).abs() < 1e-12, "glu[0]: {}", o[0]);
    assert!((o[1] - 2.0 * s4).abs() < 1e-12, "glu[1]: {}", o[1]);

    // grad of sum(glu): d/da_i = sigma(b_i); d/db_i = a_i * sigma(b_i)(1-sigma(b_i)).
    out.backward().expect("run backward");
    let g = input.grad().expect("glu input grad");
    let gs = g.as_slice();
    let expected = [
        s3,                    // d/d input[0] (a0)
        s4,                    // d/d input[1] (a1)
        1.0 * s3 * (1.0 - s3), // d/d input[2] (b0)
        2.0 * s4 * (1.0 - s4), // d/d input[3] (b1)
    ];
    for (i, (&got, &want)) in gs.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "glu grad[{i}]: {got} vs {want}");
    }
}

#[test]
fn test_glu_module_matches_function() {
    // The GLU module (parameter-free) must forward identically to the `glu` function.
    let data = [0.5f64, -1.0, 2.0, 0.25, 3.0, -0.5];
    let input = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([6], &data).expect("construct tensor"), true).expect("construct variable");
    let module = GLU::new(0);
    assert!(Module::<f64, MoiraiBackend>::parameters(&module).is_empty());

    let via_module = module.forward(&input).expect("run forward");
    let via_fn = glu(&input, 0).expect("run operation");
    assert_eq!(via_module.tensor.shape(), via_fn.tensor.shape());
    for (i, (&m, &f)) in via_module
        .tensor
        .as_slice()
        .iter()
        .zip(via_fn.tensor.as_slice())
        .enumerate()
    {
        assert!((m - f).abs() < 1e-15, "GLU module vs fn [{i}]: {m} vs {f}");
    }
}
