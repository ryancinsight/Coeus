use coeus_autograd::Var;
use coeus_nn::{
    binary_cross_entropy, huber_loss, nll_loss, GroupNorm, InstanceNorm1d, InstanceNorm2d, Linear,
    Module, ModuleExt, Sequential, StaticSeq,
};
use coeus_tensor::Tensor;

type B = coeus_core::MoiraiBackend;

#[test]
fn test_groupnorm_forward_backward() {
    let backend = B::default();
    let num_features = 4;
    let gn = GroupNorm::<f64, B, 2>::new(num_features, 1e-5).expect("construct module");

    // N=2, C=4, L=3
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -1.0, -2.0, -3.0, -4.0,
        -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
    ];
    let input = Var::new(
        Tensor::from_slice_on([2, 4, 3], &input_data, &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = gn.forward(&input).expect("run forward");

    assert_eq!(output.tensor.shape(), &[2, 4, 3]);

    // Backward pass
    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(input.grad().is_some());
    assert!(gn.weight.grad().is_some());
    assert!(gn.bias.grad().is_some());
}

#[test]
fn test_instancenorm1d_forward_backward() {
    let backend = B::default();
    let num_features = 3;
    let in1d = InstanceNorm1d::<f64, B>::new(num_features, 1e-5).expect("construct module");

    // N=2, C=3, L=4
    let input_data = (1..=24).map(|x| x as f64).collect::<Vec<_>>();
    let input = Var::new(
        Tensor::from_slice_on([2, 3, 4], &input_data, &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = in1d.forward(&input).expect("run forward");

    assert_eq!(output.tensor.shape(), &[2, 3, 4]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(input.grad().is_some());
    assert!(in1d.weight.grad().is_some());
    assert!(in1d.bias.grad().is_some());
}

#[test]
fn test_instancenorm2d_forward_backward() {
    let backend = B::default();
    let num_features = 2;
    let in2d = InstanceNorm2d::<f64, B>::new(num_features, 1e-5).expect("construct module");

    // N=1, C=2, H=3, W=3
    let input_data = (1..=18).map(|x| x as f64).collect::<Vec<_>>();
    let input = Var::new(
        Tensor::from_slice_on([1, 2, 3, 3], &input_data, &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = in2d.forward(&input).expect("run forward");

    assert_eq!(output.tensor.shape(), &[1, 2, 3, 3]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(input.grad().is_some());
    assert!(in2d.weight.grad().is_some());
    assert!(in2d.bias.grad().is_some());
}

#[test]
fn test_sequential_chaining() {
    let backend = B::default();
    let mut seq = Sequential::<f64, B>::new();
    seq.add(Linear::new(4, 3, true).expect("construct module"));
    seq.add(Linear::new(3, 2, true).expect("construct module"));

    let input = Var::new(Tensor::ones_on([2, 4], &backend).expect("construct tensor"), true).expect("construct variable");
    let output = seq.forward(&input).expect("run forward");

    assert_eq!(output.tensor.shape(), &[2, 2]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(input.grad().is_some());
    let params = seq.parameters();
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
    for p in &params {
        assert!(p.grad().is_some());
    }
}

#[test]
fn test_binary_cross_entropy_loss() {
    let backend = B::default();
    let pred = Var::new(Tensor::from_slice_on([3], &[0.1, 0.9, 0.5], &backend).expect("construct tensor"), true).expect("construct variable");
    let target = Var::new(
        Tensor::from_slice_on([3], &[0.0, 1.0, 0.0], &backend).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let loss = binary_cross_entropy(&pred, &target, 1e-7).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward().expect("run backward");
    assert!(pred.grad().is_some());
}

#[test]
fn test_nll_loss() {
    let backend = B::default();
    let log_probs = Var::new(
        Tensor::from_slice_on([2, 3], &[-0.5, -2.0, -1.5, -1.0, -0.2, -3.0], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let targets = vec![0, 1];

    let loss = nll_loss(&log_probs, &targets).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward().expect("run backward");
    assert!(log_probs.grad().is_some());
}

#[test]
fn test_huber_loss() {
    let backend = B::default();
    let pred = Var::new(Tensor::from_slice_on([3], &[1.0, 3.0, 1.5], &backend).expect("construct tensor"), true).expect("construct variable");
    let target = Var::new(
        Tensor::from_slice_on([3], &[1.5, 2.0, 4.0], &backend).expect("construct tensor"),
        false,
    ).expect("construct variable");

    let loss = huber_loss(&pred, &target, 1.0).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward().expect("run backward");
    assert!(pred.grad().is_some());
}

#[test]
fn test_static_sequential_chaining() {
    let backend = B::default();
    let model: StaticSeq<Linear<f64, B>, Linear<f64, B>> =
        Linear::<f64, B>::new(4, 3, true).expect("construct module").append(Linear::<f64, B>::new(3, 2, true).expect("construct module"));

    let input = Var::new(Tensor::ones_on([2, 4], &backend).expect("construct tensor"), true).expect("construct variable");
    let output = model.forward(&input).expect("run forward");

    assert_eq!(output.tensor.shape(), &[2, 2]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(input.grad().is_some());
    let params = model.parameters();
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
    for p in &params {
        assert!(p.grad().is_some());
    }
}
