use coeus_autograd::Var;
use coeus_nn::{init, layer_norm, rms_norm, LayerNorm, Module, ModuleError, RMSNorm};
use coeus_tensor::Tensor;

#[test]
fn test_layernorm() {
    let mut ln = LayerNorm::<f64>::new(4, 1e-5);
    init::constant(&mut ln.weight, 1.0);
    init::constant(&mut ln.bias, 0.0);

    let input = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
        true,
    );

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
    assert!(input.grad().is_some());
    assert!(ln.weight.grad().is_some());
    assert!(ln.bias.grad().is_some());
}

#[test]
fn layernorm_rejects_invalid_rank_with_typed_error() {
    let layer = LayerNorm::<f64>::new(4, 1e-5);
    let input = Var::new(Tensor::zeros(vec![4]), false);

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
    let layer = LayerNorm::<f64>::new(4, 1e-5);
    let input = Var::new(
        Tensor::from_slice([2, 1, 4], &[1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]),
        true,
    );

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
fn test_rmsnorm() {
    let mut rms = RMSNorm::<f64>::new(3, 1e-5);
    init::constant(&mut rms.weight, 1.0);

    let input = Var::new(Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]), true);
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
    assert!(input.grad().is_some());
    assert!(rms.weight.grad().is_some());
}

#[test]
fn rmsnorm_rejects_invalid_rank_with_typed_error() {
    let layer = RMSNorm::<f64>::new(4, 1e-5);
    let input = Var::new(Tensor::zeros(vec![4]), false);

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
        let mut ln = LayerNorm::<f64>::new(dim, 1e-5);
        init::constant(&mut ln.weight, 1.0);
        init::constant(&mut ln.bias, 0.0);

        let mut data = Vec::with_capacity(batch * dim);
        for i in 0..(batch * dim) {
            data.push((i + 1) as f64);
        }
        let input = Var::new(Tensor::from_slice(vec![batch, dim], &data), true);

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
        assert!(input.grad().is_some());
        assert!(ln.weight.grad().is_some());
        assert!(ln.bias.grad().is_some());
        assert_eq!(Module::<f64>::parameters(&ln).len(), 2);
    }
}

#[test]
fn test_layernorm_single_element() {
    let mut ln = LayerNorm::<f64>::new(1, 1e-5);
    init::constant(&mut ln.weight, 2.0);
    init::constant(&mut ln.bias, 1.0);

    let input = Var::new(Tensor::from_slice(vec![2, 1], &[3.0f64, 5.0]), true);
    let output = ln.forward(&input).expect("valid LayerNorm input");

    assert_eq!(output.tensor.shape(), &[2, 1]);
    let s = output.tensor.as_slice();
    assert!((s[0] - 1.0).abs() < 1e-3);
    assert!((s[1] - 1.0).abs() < 1e-3);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
}
