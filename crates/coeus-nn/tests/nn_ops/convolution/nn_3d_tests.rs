use coeus_autograd::Var;
use coeus_nn::{AvgPool3d, BatchNorm3d, Conv3d, MaxPool3d, Module};
use coeus_tensor::Tensor;

#[test]
fn test_conv3d_comprehensive() {
    let mut conv = Conv3d::<f64>::with_params(1, 1, 2, 1, 0, 1, true);

    // Set custom weight values to verify lookup values
    // Weight shape: [1, 1, 2, 2, 2]
    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    conv.weight.tensor = Tensor::from_slice(vec![1, 1, 2, 2, 2], &w_data);

    if let Some(ref mut bias) = conv.bias {
        bias.tensor = Tensor::from_slice(vec![1], &[0.5]);
    }

    // Construct a non-contiguous input tensor of shape [1, 1, 2, 2, 2]
    // by slicing a larger contiguous tensor.
    let raw_input = Tensor::from_slice(
        vec![1, 1, 2, 2, 3],
        &[
            1.0, 2.0, 999.0, 3.0, 4.0, 999.0, 5.0, 6.0, 999.0, 7.0, 8.0, 999.0,
        ],
    );
    let sliced_input = raw_input.slice(&[(0, 1), (0, 1), (0, 2), (0, 2), (0, 2)]);
    assert!(!sliced_input.is_contiguous());

    let input = Var::new(sliced_input, true);

    // Forward pass
    let output = conv.forward(&input);

    // Expected output shape: [1, 1, 1, 1, 1]
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);

    // Expected output value: sum(input_i * weight_i) + bias
    // input_i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    // weight_i = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    // sum = 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*7 + 8*8 = 204.0
    // sum + bias = 204.0 + 0.5 = 204.5
    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice.len(), 1);
    assert!((out_slice[0] - 204.5).abs() < 1e-7);

    // Backward pass
    output.backward();

    // Verify gradients
    // dy/dx_i = w_i
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad.shape(), &[1, 1, 2, 2, 2]);
    let in_g_slice = input_grad.as_slice();
    for i in 0..8 {
        assert!((in_g_slice[i] - w_data[i]).abs() < 1e-7);
    }

    // dy/dw_i = x_i
    let weight_grad = conv.weight.grad().unwrap();
    assert_eq!(weight_grad.shape(), &[1, 1, 2, 2, 2]);
    let w_g_slice = weight_grad.as_slice();
    let expected_input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for i in 0..8 {
        assert!((w_g_slice[i] - expected_input_data[i]).abs() < 1e-7);
    }

    // dy/db = 1.0
    if let Some(ref bias) = conv.bias {
        let bias_grad = bias.grad().unwrap();
        assert_eq!(bias_grad.shape(), &[1]);
        assert!((bias_grad.as_slice()[0] - 1.0).abs() < 1e-7);
    }
}

#[test]
fn test_max_pool3d_comprehensive() {
    let pool = MaxPool3d::<f64>::with_params(2, 2, 0, 1);

    // Construct a non-contiguous input tensor of shape [1, 1, 2, 2, 2]
    // with values 1 to 8.
    let raw_input = Tensor::from_slice(
        vec![1, 1, 2, 2, 3],
        &[
            1.0, 2.0, -999.0, 3.0, 4.0, -999.0, 5.0, 6.0, -999.0, 7.0, 8.0, -999.0,
        ],
    );
    let sliced_input = raw_input.slice(&[(0, 1), (0, 1), (0, 2), (0, 2), (0, 2)]);
    assert!(!sliced_input.is_contiguous());

    let input = Var::new(sliced_input, true);

    // Forward pass
    let output = pool.forward(&input);

    // Expected output shape: [1, 1, 1, 1, 1]
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    // Expected output value: max(1..8) = 8.0
    assert_eq!(output.tensor.as_slice(), &[8.0]);

    // Backward pass
    output.backward();

    // Verify gradients
    // dy/dx should be 1.0 at index 7 (where element is 8.0) and 0.0 elsewhere
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad.shape(), &[1, 1, 2, 2, 2]);
    let in_g_slice = input_grad.as_slice();
    for &val in in_g_slice.iter().take(7) {
        assert_eq!(val, 0.0);
    }
    assert_eq!(in_g_slice[7], 1.0);
}

#[test]
fn test_avg_pool3d_comprehensive() {
    let pool = AvgPool3d::<f64>::with_params(2, 2, 0, 1);

    // Construct a non-contiguous input tensor of shape [1, 1, 2, 2, 2]
    // with values 1 to 8.
    let raw_input = Tensor::from_slice(
        vec![1, 1, 2, 2, 3],
        &[
            1.0, 2.0, -999.0, 3.0, 4.0, -999.0, 5.0, 6.0, -999.0, 7.0, 8.0, -999.0,
        ],
    );
    let sliced_input = raw_input.slice(&[(0, 1), (0, 1), (0, 2), (0, 2), (0, 2)]);
    assert!(!sliced_input.is_contiguous());

    let input = Var::new(sliced_input, true);

    // Forward pass
    let output = pool.forward(&input);

    // Expected output shape: [1, 1, 1, 1, 1]
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    // Expected output value: mean(1..8) = 36 / 8 = 4.5
    assert_eq!(output.tensor.as_slice(), &[4.5]);

    // Backward pass
    output.backward();

    // Verify gradients
    // dy/dx_i should be 1/8 = 0.125 for all i
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad.shape(), &[1, 1, 2, 2, 2]);
    let in_g_slice = input_grad.as_slice();
    for &val in in_g_slice.iter().take(8) {
        assert!((val - 0.125).abs() < 1e-7);
    }
}

#[test]
fn test_batchnorm3d_comprehensive() {
    let bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1);

    // Construct a non-contiguous input tensor of shape [1, 2, 2, 2, 2]
    // Channel 0 has all 1.0s, Channel 1 has all 2.0s
    let raw_input = Tensor::from_slice(
        vec![1, 2, 2, 2, 3],
        &[
            // channel 0
            1.0, 1.0, 999.0, 1.0, 1.0, 999.0, 1.0, 1.0, 999.0, 1.0, 1.0, 999.0,
            // channel 1
            2.0, 2.0, 999.0, 2.0, 2.0, 999.0, 2.0, 2.0, 999.0, 2.0, 2.0, 999.0,
        ],
    );
    let sliced_input = raw_input.slice(&[(0, 1), (0, 2), (0, 2), (0, 2), (0, 2)]);
    assert!(!sliced_input.is_contiguous());

    let input = Var::new(sliced_input, true);

    // Forward pass
    let output = bn.forward(&input);

    // Expected output shape: [1, 2, 2, 2, 2]
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2, 2]);

    // Because elements in each channel are constant, their variance is 0.
    // Normalized value x_hat is 0.0.
    // y = gamma * x_hat + beta = 1.0 * 0.0 + 0.0 = 0.0.
    let out_slice = output.tensor.as_slice();
    for &val in out_slice {
        assert!(val.abs() < 1e-7);
    }

    // Verify running stats updates:
    // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    // For channel 0: batch_mean = 1.0, momentum = 0.1. New mean = 0.9 * 0.0 + 0.1 * 1.0 = 0.1.
    // For channel 1: batch_mean = 2.0, momentum = 0.1. New mean = 0.9 * 0.0 + 0.1 * 2.0 = 0.2.
    // running_var = (1 - momentum) * running_var + momentum * batch_var_unbiased
    // For both: batch_var_unbiased = 0.0. New var = 0.9 * 1.0 + 0.1 * 0.0 = 0.9.
    {
        let rm = bn.running_mean.borrow();
        let rv = bn.running_var.borrow();
        assert!((rm.as_slice()[0] - 0.1).abs() < 1e-7);
        assert!((rm.as_slice()[1] - 0.2).abs() < 1e-7);
        assert!((rv.as_slice()[0] - 0.9).abs() < 1e-7);
        assert!((rv.as_slice()[1] - 0.9).abs() < 1e-7);
    }

    // Backward pass
    output.backward();

    // Verify gradients exist on inputs, weight, and bias
    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());

    let w_grad = bn.weight.grad().unwrap();
    let b_grad = bn.bias.grad().unwrap();
    assert_eq!(w_grad.shape(), &[2]);
    assert_eq!(b_grad.shape(), &[2]);
}
