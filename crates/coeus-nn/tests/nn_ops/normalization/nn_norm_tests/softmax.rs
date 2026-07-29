use coeus_autograd::Var;
use coeus_nn::{softmax, Module, ModuleError, Softmax};
use coeus_tensor::Tensor;

#[test]
fn test_softmax_forward_shapes() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        true,
    );

    let output = softmax(&input, -1);
    assert_eq!(output.tensor.shape(), &[2, 3]);
}

#[test]
fn test_softmax_sums_to_one() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    let s = output.tensor.as_slice();
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(s[0] < s[1]);
    assert!(s[1] < s[2]);
}

#[test]
fn test_softmax_backward_uniform_seed() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());

    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);
    for &v in g.as_slice() {
        assert!((v - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_softmax_backward_nonuniform_seed() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    let seed = Tensor::from_slice(vec![1, 3], &[1.0, 0.0, 0.0]);
    output
        .backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");

    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);

    let y = output.tensor.as_slice();
    let g_slice = g.as_slice();

    let expected_dx0 = y[0] * (1.0 - y[0]);
    let expected_dx1 = -y[1] * y[0];
    let expected_dx2 = -y[2] * y[0];

    assert!((g_slice[0] - expected_dx0).abs() < 1e-6);
    assert!((g_slice[1] - expected_dx1).abs() < 1e-6);
    assert!((g_slice[2] - expected_dx2).abs() < 1e-6);

    assert!(g_slice[0].abs() > 1e-10);
    assert!(g_slice[1].abs() > 1e-10);
}

#[test]
fn test_softmax_module() {
    let sm = Softmax::new(-1);
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]),
        true,
    );

    let output = sm.forward(&input).expect("valid Softmax input");
    assert_eq!(output.tensor.shape(), &[2, 4]);
    assert_eq!(Module::<f64>::parameters(&sm).len(), 0);

    let s = output.tensor.as_slice();
    let row0_sum: f64 = s[0..4].iter().sum();
    let row1_sum: f64 = s[4..8].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_stability() {
    let sm = Softmax::new(-1);
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[800.0, 801.0, 802.0]), true);

    let output = sm.forward(&input).expect("valid Softmax input");
    let s = output.tensor.as_slice();

    assert!(!s[0].is_nan() && !s[0].is_infinite());
    assert!(!s[1].is_nan() && !s[1].is_infinite());
    assert!(!s[2].is_nan() && !s[2].is_infinite());

    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    assert!((s[0] - 0.09003).abs() < 1e-4);
    assert!((s[1] - 0.244728).abs() < 1e-4);
    assert!((s[2] - 0.66524).abs() < 1e-4);
}

#[test]
fn softmax_rejects_axes_outside_rank() {
    let input: Var<f64> = Var::new(Tensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]), false);

    for dim in [2, -3] {
        let error = Softmax::new(dim)
            .forward(&input)
            .err()
            .expect("Softmax axis outside rank must be rejected");
        match error {
            ModuleError::InvalidAxis { module, axis, rank } => {
                assert_eq!(module, "Softmax");
                assert_eq!(axis, dim.unsigned_abs());
                assert_eq!(rank, 2);
            }
            other => panic!("expected typed Softmax axis error, got {other:?}"),
        }
    }
}
