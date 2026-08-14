#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::Var;
use coeus_nn::{AvgPool2d, MaxPool2d, Module, ModuleError};
use coeus_tensor::Tensor;

#[test]
fn test_avg_pool2d_forward_backward() {
    let pool = AvgPool2d::<f64>::with_params(2, 2, 0, 1);
    let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 1, 4, 4], &input_data), true);
    let output = pool.forward(&input).expect("valid MaxPool2d input");

    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);
    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice, &[3.5, 5.5, 11.5, 13.5]);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    for &val in grad.as_slice() {
        assert_eq!(val, 0.25);
    }
}

#[test]
fn test_max_pool2d_forward_backward() {
    let pool = MaxPool2d::<f64>::with_params(2, 2, 0, 1);
    let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 1, 4, 4], &input_data), true);
    let output = pool.forward(&input).expect("valid AvgPool2d input");

    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);
    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice, &[6.0, 8.0, 14.0, 16.0]);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    let grad_slice = grad.as_slice();

    #[expect(clippy::needless_range_loop, reason = "ratchet COEUS-LINT-1")]
    for i in 0..16 {
        if i == 5 || i == 7 || i == 13 || i == 15 {
            assert_eq!(grad_slice[i], 1.0);
        } else {
            assert_eq!(grad_slice[i], 0.0);
        }
    }
}

#[test]
fn pool2d_rejects_invalid_window_configuration() {
    let input: Var<f64> = Var::new(Tensor::ones([1, 1, 4, 4]), false);
    for error in [
        AvgPool2d::<f64>::new(0)
            .forward(&input)
            .err()
            .expect("zero AvgPool2d kernel must be rejected"),
        MaxPool2d::<f64>::with_params(2, 0, 0, 1)
            .forward(&input)
            .err()
            .expect("zero MaxPool2d stride must be rejected"),
    ] {
        match error {
            ModuleError::ShapeMismatch {
                parameter, actual, ..
            } => {
                assert_eq!(parameter, "pooling window");
                assert!(actual.contains(&0));
            }
            other => panic!("expected typed Pool2d configuration error, got {other:?}"),
        }
    }
}
