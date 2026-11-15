use coeus_autograd::Var;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn test_groupnorm_forward_and_backward() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    // G=2, C=4, N=1, L=3: each group normalizes over C/G*L = 6 elements.
    let eps = 1e-5_f64;
    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 2>::new(4, eps)
        .expect("invariant: test operation succeeds");
    let input_data = [
        1.0f64, 2.0, 3.0, // ch 0
        4.0, 5.0, 6.0, // ch 1
        7.0, 8.0, 9.0, // ch 2
        10.0, 11.0, 12.0, // ch 3
    ];
    let input = Var::new(
        Tensor::from_slice(vec![1, 4, 3], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = gn.forward(&input).expect("valid GroupNorm input");
    assert_eq!(output.tensor.shape(), &[1, 4, 3]);

    // Wu & He (2018): x_hat = (x - mu_g) / sqrt(var_g + eps), affine weight=1, bias=0.
    let group0 = &input_data[0..6];
    let mean0: f64 = group0.iter().sum::<f64>() / 6.0;
    let var0: f64 = group0
        .iter()
        .map(|&value| {
            let delta = value - mean0;
            delta * delta
        })
        .sum::<f64>()
        / 6.0;
    let inv0 = 1.0 / (var0 + eps).sqrt();
    let group1 = &input_data[6..12];
    let mean1: f64 = group1.iter().sum::<f64>() / 6.0;
    let var1: f64 = group1
        .iter()
        .map(|&value| {
            let delta = value - mean1;
            delta * delta
        })
        .sum::<f64>()
        / 6.0;
    let inv1 = 1.0 / (var1 + eps).sqrt();
    let expected: Vec<f64> = group0
        .iter()
        .map(|&value| (value - mean0) * inv0)
        .chain(group1.iter().map(|&value| (value - mean1) * inv1))
        .collect();

    let tol = 1e-12_f64;
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "groupnorm output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // For sum loss, d/dx_i of sum_j x_hat_j within a normalized group is zero.
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    assert_eq!(input_grad.shape(), &[1, 4, 3]);
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(grad.is_finite(), "input grad[{index}] not finite: {grad}");
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for per-group normalized sum"
        );
    }

    let weight_grad = gn
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    assert_eq!(weight_grad.shape(), &[4]);
    for channel in 0..4 {
        let expected_grad: f64 = expected[channel * 3..(channel + 1) * 3].iter().sum();
        let actual = weight_grad.as_slice()[channel];
        assert!(
            (actual - expected_grad).abs() <= 1e-10,
            "weight grad[{channel}]={actual}, expected {expected_grad}"
        );
    }

    let bias_grad = gn
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.shape(), &[4]);
    assert_eq!(bias_grad.as_slice(), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_groupnorm_g1_is_layernorm() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    let eps = 1e-5_f64;
    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 1>::new(4, eps)
        .expect("invariant: test operation succeeds");
    let rows = [[1.0f64, 2.0, 3.0, 4.0], [5.0f64, 6.0, 7.0, 8.0]];
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    let input = Var::new(
        Tensor::from_slice(vec![2, 4], &flat).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = gn.forward(&input).expect("valid GroupNorm input");
    assert_eq!(output.tensor.shape(), &[2, 4]);

    // G=1 normalizes all channels together per sample, as LayerNorm over C.
    let tol = 1e-12_f64;
    let out_slice = output.tensor.as_slice();
    for (row_index, row) in rows.iter().enumerate() {
        let count = f64::from(u32::try_from(row.len()).expect("invariant: row length fits u32"));
        let mean: f64 = row.iter().sum::<f64>() / count;
        let var: f64 = row
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;
        let inv = 1.0 / (var + eps).sqrt();
        for (col, &value) in row.iter().enumerate() {
            let expected = (value - mean) * inv;
            let actual = out_slice[row_index * 4 + col];
            assert!(
                (actual - expected).abs() <= tol,
                "groupnorm g=1 output[{row_index},{col}]={actual}, expected {expected}"
            );
        }
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    assert_eq!(input_grad.shape(), &[2, 4]);
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
}

#[test]
fn test_instancenorm1d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(3, 1e-5)
        .expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros(vec![2, 3, 4]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = inst.forward(&input).expect("valid InstanceNorm1d input");
    assert_eq!(output.tensor.shape(), &[2, 3, 4]);

    // Constant input: (x - mu) = 0 for every instance/channel, so output is bias (0).
    for &value in output.tensor.as_slice() {
        assert_eq!(value, 0.0);
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    assert_eq!(input_grad.shape(), &[2, 3, 4]);
    for &grad in input_grad.as_slice() {
        assert_eq!(grad, 0.0);
    }
    let weight_grad = inst
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    assert_eq!(weight_grad.shape(), &[3]);
    assert_eq!(weight_grad.as_slice(), &[0.0, 0.0, 0.0]);
    let bias_grad = inst
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.shape(), &[3]);
    // Sum loss over [N, C, L] => d/d bias_c = N * L.
    assert_eq!(bias_grad.as_slice(), &[8.0, 8.0, 8.0]);
}

#[test]
fn test_instancenorm1d_non_constant_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    let eps = 1e-5_f64;
    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(2, eps)
        .expect("invariant: test operation succeeds");
    let input_data = [
        1.0f64, 2.0, 3.0, 4.0, // ch 0
        0.0, 0.5, 1.0, 1.5, // ch 1
    ];
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = inst.forward(&input).expect("valid InstanceNorm1d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 4]);

    let tol = 1e-12_f64;
    let mut expected = Vec::with_capacity(input_data.len());
    for channel in 0..2 {
        let slice = &input_data[channel * 4..(channel + 1) * 4];
        let mean: f64 = slice.iter().sum::<f64>() / 4.0;
        let var: f64 = slice
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / 4.0;
        let inv = 1.0 / (var + eps).sqrt();
        expected.extend(slice.iter().map(|&value| (value - mean) * inv));
    }
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "instancenorm1d output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    assert_eq!(input_grad.shape(), &[1, 2, 4]);
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(grad.is_finite(), "input grad[{index}] not finite: {grad}");
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    let weight_grad = inst
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    for channel in 0..2 {
        let expected_grad: f64 = expected[channel * 4..(channel + 1) * 4].iter().sum();
        let actual = weight_grad.as_slice()[channel];
        assert!(
            (actual - expected_grad).abs() <= 1e-10,
            "weight grad[{channel}]={actual}, expected {expected_grad}"
        );
    }
    let bias_grad = inst
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.as_slice(), &[4.0, 4.0]);
}

#[test]
fn test_instancenorm2d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm2d;

    let eps = 1e-5_f64;
    let inst = InstanceNorm2d::<f64, coeus_core::MoiraiBackend>::new(2, eps)
        .expect("invariant: test operation succeeds");
    let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 3, 3], &data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = inst.forward(&input).expect("valid InstanceNorm2d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 3, 3]);

    let tol = 1e-12_f64;
    let mut expected = Vec::with_capacity(data.len());
    for channel in 0..2 {
        let slice = &data[channel * 9..(channel + 1) * 9];
        let mean: f64 = slice.iter().sum::<f64>() / 9.0;
        let var: f64 = slice
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / 9.0;
        let inv = 1.0 / (var + eps).sqrt();
        expected.extend(slice.iter().map(|&value| (value - mean) * inv));
    }
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "instancenorm2d output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    assert_eq!(input_grad.shape(), &[1, 2, 3, 3]);
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    let weight_grad = inst
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    for channel in 0..2 {
        let expected_grad: f64 = expected[channel * 9..(channel + 1) * 9].iter().sum();
        let actual = weight_grad.as_slice()[channel];
        assert!(
            (actual - expected_grad).abs() <= 1e-10,
            "weight grad[{channel}]={actual}, expected {expected_grad}"
        );
    }
    let bias_grad = inst
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.as_slice(), &[9.0, 9.0]);
}
