use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{GroupNorm, InstanceNorm1d, InstanceNorm2d, Module};
use coeus_tensor::Tensor;

#[test]
fn test_group_norm() {
    let eps = 1e-5_f64;
    let gn = GroupNorm::<f64, MoiraiBackend, 3>::new(6, eps)
        .expect("invariant: test operation succeeds");

    let params = gn.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[6]);
    assert_eq!(params[1].tensor.shape(), &[6]);

    // [N=2, C=6, L=4], G=3 => 2 channels per group, 8 elements per (n,group).
    let input_data: Vec<f64> = (0i32..48).map(f64::from).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 6, 4], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = gn.forward(&input).expect("valid GroupNorm input");
    assert_eq!(output.tensor.shape(), &[2, 6, 4]);

    let tol = 1e-12_f64;
    let out_slice = output.tensor.as_slice();
    for n in 0..2 {
        for g in 0..3 {
            let channels = [g * 2, g * 2 + 1];
            let mut group_vals = Vec::with_capacity(8);
            for &c in &channels {
                let base = (n * 6 + c) * 4;
                group_vals.extend_from_slice(&input_data[base..base + 4]);
            }
            let mean = group_vals.iter().sum::<f64>() / 8.0;
            let var = group_vals
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / 8.0;
            let inv = 1.0 / (var + eps).sqrt();
            for &c in &channels {
                for l in 0..4 {
                    let idx = (n * 6 + c) * 4 + l;
                    let expected = (input_data[idx] - mean) * inv;
                    let actual = out_slice[idx];
                    assert!(
                        (actual - expected).abs() <= tol,
                        "group_norm output[{idx}]={actual}, expected {expected}"
                    );
                }
            }
        }
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    assert_eq!(gn.weight.grad().expect("weight grad").shape(), &[6]);
    assert_eq!(gn.bias.grad().expect("bias grad").shape(), &[6]);
    assert_eq!(
        gn.bias.grad().expect("bias grad").as_slice(),
        &[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    );
}

#[test]
fn test_instance_norm_1d() {
    let eps = 1e-5_f64;
    let in1d = InstanceNorm1d::<f64, MoiraiBackend>::new(4, eps)
        .expect("invariant: test operation succeeds");

    let params = in1d.parameters();
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].tensor.shape(), &[4]);
    assert_eq!(params[1].tensor.shape(), &[4]);

    // [2,4,5], each (n,c) is 5 consecutive values => mean at center, pop var 2.
    let input_data: Vec<f64> = (0i32..40).map(f64::from).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 4, 5], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = in1d.forward(&input).expect("valid InstanceNorm1d input");
    assert_eq!(output.tensor.shape(), &[2, 4, 5]);

    let inv = 1.0 / (2.0 + eps).sqrt();
    let tol = 1e-12_f64;
    let out_slice = output.tensor.as_slice();
    for n in 0..2 {
        for c in 0..4 {
            let base = (n * 4 + c) * 5;
            let slice = &input_data[base..base + 5];
            let mean = slice.iter().sum::<f64>() / 5.0;
            for l in 0..5 {
                let idx = base + l;
                let expected = (input_data[idx] - mean) * inv;
                let actual = out_slice[idx];
                assert!(
                    (actual - expected).abs() <= tol,
                    "instance_norm1d output[{idx}]={actual}, expected {expected}"
                );
            }
        }
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input.grad().expect("input grad");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0"
        );
    }
    assert_eq!(
        in1d.bias.grad().expect("bias grad").as_slice(),
        &[10.0, 10.0, 10.0, 10.0]
    );
}

#[test]
fn test_instance_norm_2d() {
    let eps = 1e-5_f64;
    let in2d = InstanceNorm2d::<f64, MoiraiBackend>::new(3, eps)
        .expect("invariant: test operation succeeds");

    let params = in2d.parameters();
    assert_eq!(params.len(), 2);

    // [2,3,4,4], each (n,c) is 16 consecutive values => pop var (16^2-1)/12 = 21.25.
    let input_data: Vec<f64> = (0i32..96).map(f64::from).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3, 4, 4], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = in2d.forward(&input).expect("valid InstanceNorm2d input");
    assert_eq!(output.tensor.shape(), &[2, 3, 4, 4]);

    let inv = 1.0 / (21.25 + eps).sqrt();
    let tol = 1e-12_f64;
    let out_slice = output.tensor.as_slice();
    for n in 0..2 {
        for c in 0..3 {
            let base = (n * 3 + c) * 16;
            let slice = &input_data[base..base + 16];
            let mean = slice.iter().sum::<f64>() / 16.0;
            for k in 0..16 {
                let idx = base + k;
                let expected = (input_data[idx] - mean) * inv;
                let actual = out_slice[idx];
                assert!(
                    (actual - expected).abs() <= tol,
                    "instance_norm2d output[{idx}]={actual}, expected {expected}"
                );
            }
        }
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input.grad().expect("input grad");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0"
        );
    }
    assert_eq!(
        in2d.bias.grad().expect("bias grad").as_slice(),
        &[32.0, 32.0, 32.0]
    );
}
