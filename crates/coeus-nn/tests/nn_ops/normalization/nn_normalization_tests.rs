use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{GroupNorm, InstanceNorm1d, InstanceNorm2d, Module};
use coeus_tensor::Tensor;

#[test]
fn test_group_norm() {
    let gn = GroupNorm::<f64, MoiraiBackend, 3>::new(6, 1e-5).expect("construct module");

    // parameters check
    let params = gn.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[6]); // weight
    assert_eq!(params[1].tensor.shape(), &[6]); // bias

    // 3D input: [batch=2, channels=6, spatial=4]
    let input_data: Vec<f64> = (0..48).map(|x| x as f64).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 6, 4], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = gn.forward(&input).expect("run forward");
    assert_eq!(output.tensor.shape(), &[2, 6, 4]);

    // Backward pass
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    assert!(gn.weight.grad().is_some());
    assert!(gn.bias.grad().is_some());
}

#[test]
fn test_instance_norm_1d() {
    let in1d = InstanceNorm1d::<f64, MoiraiBackend>::new(4, 1e-5).expect("construct module");

    let params = in1d.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[4]);
    assert_eq!(params[1].tensor.shape(), &[4]);

    // 3D input: [batch=2, channels=4, spatial=5]
    let input_data: Vec<f64> = (0..40).map(|x| x as f64).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 4, 5], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = in1d.forward(&input).expect("run forward");
    assert_eq!(output.tensor.shape(), &[2, 4, 5]);

    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    assert!(in1d.weight.grad().is_some());
    assert!(in1d.bias.grad().is_some());
}

#[test]
fn test_instance_norm_2d() {
    let in2d = InstanceNorm2d::<f64, MoiraiBackend>::new(3, 1e-5).expect("construct module");

    let params = in2d.parameters();
    assert_eq!(params.len(), 2);

    // 4D input: [batch=2, channels=3, height=4, width=4]
    let input_data: Vec<f64> = (0..96).map(|x| x as f64).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3, 4, 4], &input_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = in2d.forward(&input).expect("run forward");
    assert_eq!(output.tensor.shape(), &[2, 3, 4, 4]);

    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    assert!(in2d.weight.grad().is_some());
    assert!(in2d.bias.grad().is_some());
}
