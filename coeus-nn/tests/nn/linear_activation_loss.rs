use coeus_autograd::Var;
use coeus_nn::{cross_entropy_loss, gelu, init, mse_loss, relu, sigmoid, tanh, Linear, Module};
use coeus_tensor::Tensor;

fn assert_slice_close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual {}, expected {}",
        actual.len(),
        expected.len()
    );

    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{label}[{index}]: actual {actual}, expected {expected}, diff {diff}, tolerance {tolerance}"
        );
    }
}

fn expected_cross_entropy_gradients(
    logits: &[f64],
    rows: usize,
    cols: usize,
    targets: &[usize],
) -> Vec<f64> {
    assert_eq!(logits.len(), rows * cols);
    assert_eq!(targets.len(), rows);

    let batch_scale = 1.0 / rows as f64;
    let mut expected = Vec::with_capacity(logits.len());
    for (row, &target) in targets.iter().enumerate() {
        assert!(
            target < cols,
            "target class {target} must be less than {cols}"
        );
        let row_start = row * cols;
        let row_logits = &logits[row_start..row_start + cols];
        let max_logit = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row_logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .sum();

        for (class, &logit) in row_logits.iter().enumerate() {
            let mut grad = ((logit - max_logit).exp() / exp_sum) * batch_scale;
            if class == target {
                grad -= batch_scale;
            }
            expected.push(grad);
        }
    }

    expected
}

#[test]
fn test_linear_layer() {
    let mut layer = Linear::<f64>::new(3, 2, true);
    init::constant(&mut layer.weight, 1.0);
    if let Some(ref mut b) = layer.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]), true);
    let output = layer.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 2]);
    assert_eq!(output.tensor.as_slice(), &[6.5, 6.5]);

    output.backward();
    assert_eq!(
        input
            .grad()
            .expect("linear input gradient must be populated")
            .as_slice(),
        &[2.0, 2.0, 2.0]
    );
    assert_eq!(
        layer
            .weight
            .grad()
            .expect("linear weight gradient must be populated")
            .as_slice(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
    if let Some(ref b) = layer.bias {
        assert_eq!(
            b.grad()
                .expect("linear bias gradient must be populated")
                .as_slice(),
            &[1.0, 1.0]
        );
    }
}

#[test]
fn test_activations() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![4], &[-2.0f64, -0.5, 0.5, 2.0]),
        true,
    );

    // ReLU
    let out_relu = relu(&input);
    assert_eq!(out_relu.tensor.as_slice(), &[0.0, 0.0, 0.5, 2.0]);
    out_relu.backward();
    assert_eq!(input.grad().unwrap().as_slice(), &[0.0, 0.0, 1.0, 1.0]);

    // Sigmoid
    input.zero_grad();
    let out_sig = sigmoid(&input);
    assert!((out_sig.tensor.as_slice()[2] - 0.622459f64).abs() < 1e-4);

    // Tanh
    input.zero_grad();
    let out_tanh = tanh(&input);
    assert!((out_tanh.tensor.as_slice()[2] - 0.462117f64).abs() < 1e-4);

    // GeLU
    input.zero_grad();
    let out_gelu = gelu(&input);
    assert!(out_gelu.tensor.as_slice()[0] < 0.1);
}

#[test]
fn test_losses() {
    let pred: Var<f64> = Var::new(Tensor::from_slice(vec![2], &[0.5f64, 1.5]), true);
    let target = Var::new(Tensor::from_slice(vec![2], &[1.0f64, 1.0]), false);

    // MSE
    let loss_mse = mse_loss(&pred, &target);
    assert_eq!(loss_mse.tensor.as_slice(), &[0.25]);
    loss_mse.backward();
    assert_eq!(
        pred.grad()
            .expect("mse prediction gradient must be populated")
            .as_slice(),
        &[-0.5, 0.5]
    );

    // Cross entropy
    let logits_values = &[1.0f64, 2.0, 0.0, 0.0, 2.0, 1.0];
    let logits: Var<f64> = Var::new(Tensor::from_slice(vec![2, 3], logits_values), true);
    let targets = vec![1, 2];
    let loss_ce = cross_entropy_loss(&logits, &targets);
    assert_eq!(loss_ce.tensor.shape(), &[1]);
    loss_ce.backward();
    let expected_ce_gradients = expected_cross_entropy_gradients(logits_values, 2, 3, &targets);
    assert_slice_close(
        "cross_entropy_logits_grad",
        logits
            .grad()
            .expect("cross-entropy logit gradient must be populated")
            .as_slice(),
        &expected_ce_gradients,
        1e-12,
    );
}

#[test]
fn test_initializers() {
    let mut weight = Var::<f64>::new(Tensor::zeros(vec![1000]), true);

    init::normal(&mut weight, 5.0, 2.0);
    let w_slice = weight.tensor.as_slice();
    let sum: f64 = w_slice.iter().sum();
    let mean = sum / w_slice.len() as f64;
    assert!((mean - 5.0).abs() < 0.2);

    init::xavier_uniform(&mut weight, 100, 100);
    let limit = (6.0f64 / 200.0).sqrt();
    for &val in weight.tensor.as_slice() {
        assert!(val >= -limit && val <= limit);
    }
}
