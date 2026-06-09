use coeus_autograd::Var;
use coeus_tensor::Tensor;

#[test]
fn test_log_softmax_probabilities() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0f64, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]),
        true,
    );
    let log_probs = coeus_autograd::log_softmax(&input, 1);
    assert_eq!(log_probs.tensor.shape(), &[2, 4]);

    let s = log_probs.tensor.as_slice();
    let row0_sum: f64 = s[..4].iter().map(|x| x.exp()).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row0_sum={row0_sum}");
    let row1_sum: f64 = s[4..].iter().map(|x| x.exp()).sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "row1_sum={row1_sum}");
}

#[test]
fn test_log_softmax_backward() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]), true);
    let log_probs = coeus_autograd::log_softmax(&input, 1);
    let target: Var<f64> =
        Var::new(Tensor::from_slice(vec![1, 3], &[0.0f64, 1.0, 0.0]), false);
    let loss = coeus_nn::loss::mse_loss(&log_probs, &target);
    loss.backward();
    assert!(input.grad().is_some());
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);
    assert!(g.as_slice().iter().any(|&v| v.abs() > 1e-7));
}

#[test]
fn test_cat_forward_shape() {
    let a = Var::<f64>::new(Tensor::zeros(vec![2, 3]), true);
    let b = Var::<f64>::new(Tensor::zeros(vec![2, 4]), true);
    let out = coeus_autograd::cat(&[&a, &b], 1);
    assert_eq!(out.tensor.shape(), &[2, 7]);
}

#[test]
fn test_cat_backward_gradient_split() {
    let a_data = vec![1.0f64; 6];
    let b_data = vec![2.0f64; 8];
    let a = Var::<f64>::new(Tensor::from_slice(vec![2, 3], &a_data), true);
    let b = Var::<f64>::new(Tensor::from_slice(vec![2, 4], &b_data), true);
    let out = coeus_autograd::cat(&[&a, &b], 1);
    out.backward();
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga.shape(), &[2, 3]);
    assert_eq!(gb.shape(), &[2, 4]);
}

#[test]
fn test_cat_along_dim0() {
    let a = Var::<f64>::new(Tensor::zeros(vec![2, 5]), true);
    let b = Var::<f64>::new(Tensor::zeros(vec![3, 5]), true);
    let c = Var::<f64>::new(Tensor::zeros(vec![1, 5]), true);
    let out = coeus_autograd::cat(&[&a, &b, &c], 0);
    assert_eq!(out.tensor.shape(), &[6, 5]);
    out.backward();
    assert_eq!(a.grad().unwrap().shape(), &[2, 5]);
    assert_eq!(b.grad().unwrap().shape(), &[3, 5]);
    assert_eq!(c.grad().unwrap().shape(), &[1, 5]);
}

#[test]
fn test_split_even_chunks() {
    let input = Var::<f64>::new(Tensor::zeros(vec![1, 6]), true);
    let chunks = coeus_autograd::split(&input, 2, 1);
    assert_eq!(chunks.len(), 3);
    for ch in &chunks {
        assert_eq!(ch.tensor.shape(), &[1, 2]);
    }
}

#[test]
fn test_split_remainder_chunk() {
    let input = Var::<f64>::new(Tensor::zeros(vec![1, 7]), true);
    let chunks = coeus_autograd::split(&input, 3, 1);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[1].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[2].tensor.shape(), &[1, 1]);
}

#[test]
fn test_split_backward_accumulation() {
    let input = Var::<f64>::new(
        Tensor::from_slice(vec![1, 4], &[1.0f64, 2.0, 3.0, 4.0]),
        true,
    );
    let chunks = coeus_autograd::split(&input, 2, 1);
    let target = Var::<f64>::new(Tensor::from_slice(vec![1, 2], &[0.0f64, 0.0]), false);
    let loss = coeus_nn::loss::mse_loss(&chunks[0], &target);
    loss.backward();
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 4]);
    assert_eq!(g.as_slice()[2], 0.0);
    assert_eq!(g.as_slice()[3], 0.0);
    assert!(g.as_slice()[0].abs() > 0.0 || g.as_slice()[1].abs() > 0.0);
}
