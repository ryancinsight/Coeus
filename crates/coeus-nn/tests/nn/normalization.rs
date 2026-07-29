use coeus_autograd::Var;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn test_groupnorm_forward_and_backward() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    // G=2, C=4: each group contains 2 channels.
    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 2>::new(4, 1e-5);
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 4, 3],
            &[
                1.0f64, 2.0, 3.0, // ch 0
                4.0, 5.0, 6.0, // ch 1
                7.0, 8.0, 9.0, // ch 2
                10.0, 11.0, 12.0, // ch 3
            ],
        ),
        true,
    );
    let output = gn.forward(&input).expect("valid GroupNorm input");
    assert_eq!(output.tensor.shape(), &[1, 4, 3]);

    let out_slice = output.tensor.as_slice();
    let group0_sum: f64 = out_slice[..6].iter().sum();
    assert!(group0_sum.abs() < 1e-5, "group0_sum={group0_sum}");

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    assert!(gn.weight.grad().is_some());
    assert!(gn.bias.grad().is_some());
}

#[test]
fn test_groupnorm_g1_is_layernorm() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 1>::new(4, 1e-5);
    let input = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        true,
    );
    let output = gn.forward(&input).expect("valid GroupNorm input");
    assert_eq!(output.tensor.shape(), &[2, 4]);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
}

#[test]
fn test_instancenorm1d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(3, 1e-5);
    let input = Var::new(Tensor::zeros(vec![2, 3, 4]), true);
    let output = inst.forward(&input).expect("valid InstanceNorm1d input");
    assert_eq!(output.tensor.shape(), &[2, 3, 4]);

    // All-zero input → output is all-zero
    for &v in output.tensor.as_slice() {
        assert!(v.abs() < 1e-5);
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    assert!(inst.weight.grad().is_some());
    assert!(inst.bias.grad().is_some());
}

#[test]
fn test_instancenorm1d_non_constant_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(2, 1e-5);
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 2, 4],
            &[
                1.0f64, 2.0, 3.0, 4.0, // ch 0 → mean=2.5
                0.0, 0.5, 1.0, 1.5, // ch 1 → mean=0.75
            ],
        ),
        true,
    );
    let output = inst.forward(&input).expect("valid InstanceNorm1d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 4]);

    let s = output.tensor.as_slice();
    let mean0: f64 = s[..4].iter().sum::<f64>() / 4.0;
    assert!(mean0.abs() < 1e-5);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
}

#[test]
fn test_instancenorm2d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm2d;

    let inst = InstanceNorm2d::<f64, coeus_core::MoiraiBackend>::new(2, 1e-5);
    let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 2, 3, 3], &data), true);
    let output = inst.forward(&input).expect("valid InstanceNorm2d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 3, 3]);

    let s = output.tensor.as_slice();
    let mean0: f64 = s[..9].iter().sum::<f64>() / 9.0;
    let mean1: f64 = s[9..].iter().sum::<f64>() / 9.0;
    assert!(mean0.abs() < 1e-5, "mean0={mean0}");
    assert!(mean1.abs() < 1e-5, "mean1={mean1}");

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    assert!(inst.weight.grad().is_some());
    assert!(inst.bias.grad().is_some());
}
