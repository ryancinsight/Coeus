use coeus_autograd::Var;
use coeus_tensor::Tensor;

#[test]
fn test_log_softmax_probabilities() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0f64, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let log_probs =
        coeus_autograd::log_softmax(&input, 1).expect("invariant: test operation succeeds");
    assert_eq!(log_probs.tensor.shape(), &[2, 4]);

    let s = log_probs.tensor.as_slice();
    let row0_sum: f64 = s[..4].iter().map(|x| x.exp()).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row0_sum={row0_sum}");
    let row1_sum: f64 = s[4..].iter().map(|x| x.exp()).sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "row1_sum={row1_sum}");
}

#[test]
fn test_log_softmax_backward() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let log_probs =
        coeus_autograd::log_softmax(&input, 1).expect("invariant: test operation succeeds");
    let target: Var<f64> = Var::new(
        Tensor::from_slice(vec![1, 3], &[0.0f64, 1.0, 0.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let loss =
        coeus_nn::loss::mse_loss(&log_probs, &target).expect("valid mean squared error inputs");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = input
        .grad()
        .expect("invariant: log softmax gives its input a gradient");
    assert_eq!(g.shape(), &[1, 3]);
    assert!(g.as_slice().iter().any(|&v| v.abs() > 1e-7));

    // Reference: y = log_softmax(x), p = softmax(x), loss = mean((y - t)^2).
    // dL/dy = (2/3)(y - t) and dL/dx_k = g_k - p_k * sum_j g_j.
    let x_data = [1.0_f64, 2.0, 3.0];
    let target_data = [0.0_f64, 1.0, 0.0];
    let log_sum_exp: f64 = x_data.iter().map(|&value| value.exp()).sum::<f64>().ln();
    let mut upstream = [0.0_f64; 3];
    for index in 0..3 {
        let log_prob = x_data[index] - log_sum_exp;
        upstream[index] = (2.0 / 3.0) * (log_prob - target_data[index]);
    }
    let upstream_sum: f64 = upstream.iter().sum();
    let mut expected = [0.0_f64; 3];
    for index in 0..3 {
        let probability = (x_data[index] - log_sum_exp).exp();
        expected[index] = upstream[index] - probability * upstream_sum;
    }
    for (index, (&actual, &value)) in g.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - value).abs() <= 1e-12,
            "log_softmax grad[{index}] = {actual}, expected {value}"
        );
    }
}

#[test]
fn test_cat_forward_shape() {
    let a = Var::<f64>::new(
        Tensor::zeros(vec![2, 3]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let b = Var::<f64>::new(
        Tensor::zeros(vec![2, 4]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let out = coeus_autograd::cat(&[&a, &b], 1).expect("invariant: test operation succeeds");
    assert_eq!(out.tensor.shape(), &[2, 7]);
}

#[test]
fn test_cat_backward_gradient_split() {
    let a_data = vec![1.0f64; 6];
    let b_data = vec![2.0f64; 8];
    let a = Var::<f64>::new(
        Tensor::from_slice(vec![2, 3], &a_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let b = Var::<f64>::new(
        Tensor::from_slice(vec![2, 4], &b_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let out = coeus_autograd::cat(&[&a, &b], 1).expect("invariant: test operation succeeds");
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let ga = a
        .grad()
        .expect("invariant: concatenation gives its first input a gradient");
    let gb = b
        .grad()
        .expect("invariant: concatenation gives its second input a gradient");
    assert_eq!(ga.shape(), &[2, 3]);
    assert_eq!(gb.shape(), &[2, 4]);
    // The upstream gradient of the concatenated output is ones, so every input
    // receives the ones gradient of its own slice.
    for (index, &value) in ga.as_slice().iter().enumerate() {
        assert!(
            (value - 1.0).abs() <= 1e-12,
            "cat grad a[{index}] = {value}, expected 1.0"
        );
    }
    for (index, &value) in gb.as_slice().iter().enumerate() {
        assert!(
            (value - 1.0).abs() <= 1e-12,
            "cat grad b[{index}] = {value}, expected 1.0"
        );
    }
}

#[test]
fn test_cat_along_dim0() {
    let a = Var::<f64>::new(
        Tensor::zeros(vec![2, 5]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let b = Var::<f64>::new(
        Tensor::zeros(vec![3, 5]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let c = Var::<f64>::new(
        Tensor::zeros(vec![1, 5]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let out = coeus_autograd::cat(&[&a, &b, &c], 0).expect("invariant: test operation succeeds");
    assert_eq!(out.tensor.shape(), &[6, 5]);
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(a.grad().unwrap().shape(), &[2, 5]);
    assert_eq!(b.grad().unwrap().shape(), &[3, 5]);
    assert_eq!(c.grad().unwrap().shape(), &[1, 5]);
}

#[test]
fn test_split_even_chunks() {
    let input = Var::<f64>::new(
        Tensor::zeros(vec![1, 6]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let chunks = coeus_autograd::split(&input, 2, 1).expect("invariant: test operation succeeds");
    assert_eq!(chunks.len(), 3);
    for ch in &chunks {
        assert_eq!(ch.tensor.shape(), &[1, 2]);
    }
}

#[test]
fn test_split_remainder_chunk() {
    let input = Var::<f64>::new(
        Tensor::zeros(vec![1, 7]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let chunks = coeus_autograd::split(&input, 3, 1).expect("invariant: test operation succeeds");
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[1].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[2].tensor.shape(), &[1, 1]);
}

#[test]
fn test_split_backward_accumulation() {
    let input = Var::<f64>::new(
        Tensor::from_slice(vec![1, 4], &[1.0f64, 2.0, 3.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let chunks = coeus_autograd::split(&input, 2, 1).expect("invariant: test operation succeeds");
    let target = Var::<f64>::new(
        Tensor::from_slice(vec![1, 2], &[0.0f64, 0.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let loss =
        coeus_nn::loss::mse_loss(&chunks[0], &target).expect("valid mean squared error inputs");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 4]);
    assert_eq!(g.as_slice()[2], 0.0);
    assert_eq!(g.as_slice()[3], 0.0);
    assert!(g.as_slice()[0].abs() > 0.0 || g.as_slice()[1].abs() > 0.0);
}
