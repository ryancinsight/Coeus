use coeus_autograd::Var;
use coeus_nn::{init, layer_norm, rms_norm, LayerNorm, Module, ModuleError, RMSNorm};
use coeus_tensor::Tensor;

#[test]
fn test_layernorm() {
    let mut ln = LayerNorm::<f64>::new(4, 1e-5).expect("invariant: test operation succeeds");
    init::constant(&mut ln.weight, 1.0).expect("invariant: test operation succeeds");
    init::constant(&mut ln.bias, 0.0).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = ln.forward(&input).expect("valid LayerNorm input");
    let output_fn = layer_norm(&input, 4, Some(&ln.weight), Some(&ln.bias), 1e-5)
        .expect("valid functional LayerNorm input");
    assert_eq!(output.tensor.shape(), &[2, 4]);
    assert_eq!(output_fn.tensor.shape(), &[2, 4]);
    for (a, b) in output
        .tensor
        .as_slice()
        .iter()
        .zip(output_fn.tensor.as_slice())
    {
        assert!(
            (a - b).abs() < 1e-10,
            "layer_norm functional parity: {a} vs {b}"
        );
    }

    // Output elements for each batch should have mean ~0 and std ~1
    let out_slice = output.tensor.as_slice();
    for i in 0..2 {
        let offset = i * 4;
        let mut mean = 0.0f64;
        for j in 0..4 {
            mean += out_slice[offset + j];
        }
        mean /= 4.0;
        assert!(mean.abs() < 1e-5);
    }

    // Test backward pass
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: LayerNorm input requires grad");
    let weight_grad = ln
        .weight
        .grad()
        .expect("invariant: LayerNorm weight requires grad");
    let bias_grad = ln
        .bias
        .grad()
        .expect("invariant: LayerNorm bias requires grad");
    assert_eq!(input_grad.shape(), &[2, 4]);
    assert_eq!(weight_grad.shape(), &[4]);
    assert_eq!(bias_grad.shape(), &[4]);

    // Upstream grad is ones: dL/dbias_i = sum_b 1 = batch_size.
    for (i, &g) in bias_grad.as_slice().iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-6,
            "LayerNorm bias grad[{i}] = {g}, expected 2.0"
        );
    }

    // dL/dweight_i = sum_b xhat_{b,i}. Both rows have the same normalized pattern.
    let xhat = [
        -1.341_640_786_499_873_8,
        -0.447_213_595_499_957_9,
        0.447_213_595_499_957_9,
        1.341_640_786_499_873_8,
    ];
    for (i, (&g, &xh)) in weight_grad.as_slice().iter().zip(xhat.iter()).enumerate() {
        let expected = 2.0 * xh;
        assert!(
            (g - expected).abs() < 1e-5,
            "LayerNorm weight grad[{i}] = {g}, expected {expected}"
        );
    }
}

#[test]
fn layernorm_rejects_invalid_rank_with_typed_error() {
    let layer = LayerNorm::<f64>::new(4, 1e-5).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros(vec![4]).expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let error = match layer.forward(&input) {
        Ok(_) => panic!("rank-one LayerNorm input must fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ModuleError::InvalidRank {
            module: "LayerNorm",
            expected: "at least 2",
            actual: 1
        }
    ));
}

#[test]
fn layernorm_module_normalizes_rank_three_trailing_dimension() {
    let layer = LayerNorm::<f64>::new(4, 1e-5).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice([2, 1, 4], &[1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = layer
        .forward(&input)
        .expect("rank-three input has the configured trailing dimension");
    assert_eq!(output.tensor.shape(), &[2, 1, 4]);
    for row in output.tensor.as_slice().chunks_exact(4) {
        let mean = row.iter().sum::<f64>() / 4.0;
        let variance = row.iter().map(|value| value * value).sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-12);
        assert!((variance - 1.0).abs() < 1e-4);
    }

    output
        .backward()
        .expect("invariant: valid LayerNorm graph completes backward");
    assert_eq!(
        input
            .grad()
            .expect("tracked LayerNorm input receives a gradient")
            .shape(),
        &[2, 1, 4]
    );
}

#[test]
fn layernorm_normalizes_multiple_trailing_dimensions_and_restores_gradients() {
    let layer =
        LayerNorm::<f64>::from_shape([2, 3], 1e-5).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(
            [2, 2, 3],
            &[
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = layer
        .forward(&input)
        .expect("configured trailing shape matches rank-three input");
    assert_eq!(output.tensor.shape(), &[2, 2, 3]);
    for window in output.tensor.as_slice().chunks_exact(6) {
        let mean = window.iter().sum::<f64>() / 6.0;
        let variance = window
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / 6.0;
        assert!(mean.abs() < 1e-12);
        assert!((variance - 1.0).abs() < 1e-4);
    }

    output
        .backward()
        .expect("invariant: multi-dimensional LayerNorm backward completes");
    assert_eq!(
        input
            .grad()
            .expect("tracked multi-dimensional input receives a gradient")
            .shape(),
        &[2, 2, 3]
    );
    let bias_grad = layer
        .bias
        .grad()
        .expect("tracked multi-dimensional bias receives a gradient");
    assert_eq!(bias_grad.shape(), &[2, 3]);
    assert!(bias_grad
        .as_slice()
        .iter()
        .all(|value| (*value - 2.0).abs() < 1e-12));
}

#[test]
fn layernorm_rejects_mismatched_trailing_shape() {
    let layer =
        LayerNorm::<f64>::from_shape([2, 3], 1e-5).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros([2, 3, 4]).expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let error = match layer.forward(&input) {
        Ok(_) => panic!("input trailing dimensions must match normalized_shape"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ModuleError::ShapeMismatch {
            parameter: "input trailing dimensions",
            ..
        }
    ));
}

#[test]
fn test_rmsnorm() {
    let mut rms = RMSNorm::<f64>::new(3, 1e-5).expect("invariant: test operation succeeds");
    init::constant(&mut rms.weight, 1.0).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = rms.forward(&input).expect("valid RMSNorm input");
    let output_fn =
        rms_norm(&input, Some(&rms.weight), 1e-5).expect("valid functional RMSNorm input");

    assert_eq!(output.tensor.shape(), &[1, 3]);
    assert_eq!(output_fn.tensor.shape(), &[1, 3]);
    for (a, b) in output
        .tensor
        .as_slice()
        .iter()
        .zip(output_fn.tensor.as_slice())
    {
        assert!(
            (a - b).abs() < 1e-10,
            "rms_norm functional parity: {a} vs {b}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: RMSNorm input requires grad");
    let weight_grad = rms
        .weight
        .grad()
        .expect("invariant: RMSNorm weight requires grad");
    assert_eq!(input_grad.shape(), &[1, 3]);
    assert_eq!(weight_grad.shape(), &[3]);

    // RMSNorm: y_i = w_i * x_i / rms, rms = sqrt(mean(x^2)+eps).
    // dL/dw_i = x_i / rms; with x=[1,2,3], rms=sqrt((1+4+9)/3 + eps)=sqrt(14/3+eps).
    let rms_val = ((1.0_f64 + 4.0 + 9.0) / 3.0 + 1e-5).sqrt();
    let expected_w_grad = [1.0 / rms_val, 2.0 / rms_val, 3.0 / rms_val];
    for (i, (&g, &expected)) in weight_grad
        .as_slice()
        .iter()
        .zip(expected_w_grad.iter())
        .enumerate()
    {
        assert!(
            (g - expected).abs() < 1e-6,
            "RMSNorm weight grad[{i}] = {g}, expected {expected}"
        );
    }
}

#[test]
fn rmsnorm_rejects_invalid_rank_with_typed_error() {
    let layer = RMSNorm::<f64>::new(4, 1e-5).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros(vec![4]).expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let error = match layer.forward(&input) {
        Ok(_) => panic!("rank-one RMSNorm input must fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ModuleError::InvalidRank {
            module: "RMSNorm",
            expected: "2",
            actual: 1
        }
    ));
}

#[test]
fn test_layernorm_various_shapes() {
    let shapes: Vec<(usize, usize)> = vec![(1, 8), (3, 16), (8, 4), (16, 32)];

    for &(batch, dim) in &shapes {
        let mut ln = LayerNorm::<f64>::new(dim, 1e-5).expect("invariant: test operation succeeds");
        init::constant(&mut ln.weight, 1.0).expect("invariant: test operation succeeds");
        init::constant(&mut ln.bias, 0.0).expect("invariant: test operation succeeds");

        let mut data = Vec::with_capacity(batch * dim);
        for i in 0..(batch * dim) {
            data.push((i + 1) as f64);
        }
        let input = Var::new(
            Tensor::from_slice(vec![batch, dim], &data)
                .expect("invariant: test backend operation succeeds"),
            true,
        )
        .expect("invariant: test backend operation succeeds");

        let output = ln.forward(&input).expect("valid LayerNorm input");
        assert_eq!(output.tensor.shape(), &[batch, dim]);

        let out_slice = output.tensor.as_slice();
        for i in 0..batch {
            let offset = i * dim;
            let mut mean = 0.0f64;
            for j in 0..dim {
                mean += out_slice[offset + j];
            }
            mean /= dim as f64;
            assert!(mean.abs() < 1e-5);
        }

        output
            .backward()
            .expect("invariant: valid autograd fixture completes backward");
        let input_grad = input
            .grad()
            .expect("invariant: LayerNorm input requires grad");
        let weight_grad = ln
            .weight
            .grad()
            .expect("invariant: LayerNorm weight requires grad");
        let bias_grad = ln
            .bias
            .grad()
            .expect("invariant: LayerNorm bias requires grad");
        assert_eq!(input_grad.shape(), &[batch, dim]);
        assert_eq!(weight_grad.shape(), &[dim]);
        assert_eq!(bias_grad.shape(), &[dim]);
        // Upstream ones => dL/dbias_i = batch.
        for (i, &g) in bias_grad.as_slice().iter().enumerate() {
            assert!(
                (g - batch as f64).abs() < 1e-6,
                "LayerNorm bias grad[{i}] = {g}, expected {}",
                batch
            );
        }
        assert_eq!(Module::<f64>::parameters(&ln).len(), 2);
    }
}

#[test]
fn test_layernorm_single_element() {
    let mut ln = LayerNorm::<f64>::new(1, 1e-5).expect("invariant: test operation succeeds");
    init::constant(&mut ln.weight, 2.0).expect("invariant: test operation succeeds");
    init::constant(&mut ln.bias, 1.0).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![2, 1], &[3.0f64, 5.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = ln.forward(&input).expect("valid LayerNorm input");

    assert_eq!(output.tensor.shape(), &[2, 1]);
    let s = output.tensor.as_slice();
    assert!((s[0] - 1.0).abs() < 1e-3);
    assert!((s[1] - 1.0).abs() < 1e-3);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: single-element LayerNorm input requires grad");
    let weight_grad = ln
        .weight
        .grad()
        .expect("invariant: single-element LayerNorm weight requires grad");
    let bias_grad = ln
        .bias
        .grad()
        .expect("invariant: single-element LayerNorm bias requires grad");
    assert_eq!(input_grad.shape(), &[2, 1]);
    assert_eq!(weight_grad.shape(), &[1]);
    assert_eq!(bias_grad.shape(), &[1]);
    // Normalizing over a single element yields xhat=0, y=bias: dy/dx=0, dy/dw=0, dy/db=1.
    for &g in input_grad.as_slice() {
        assert!(g.abs() < 1e-6, "LayerNorm single-element input grad {g}");
    }
    assert!(
        weight_grad.as_slice()[0].abs() < 1e-6,
        "LayerNorm single-element weight grad {}",
        weight_grad.as_slice()[0]
    );
    assert!(
        (bias_grad.as_slice()[0] - 2.0).abs() < 1e-6,
        "LayerNorm single-element bias grad {}",
        bias_grad.as_slice()[0]
    );
}
