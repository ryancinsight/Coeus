use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_pad_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 2], &[1.0f32, 2.0f32, 3.0f32, 4.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::pad(&x, &[(1, 1), (1, 1)], 0.0f32);
    assert_eq!(y.tensor.shape(), &[4, 4]);

    let y_slice = y.tensor.as_slice();
    assert_eq!(y_slice[5], 1.0);
    assert_eq!(y_slice[6], 2.0);
    assert_eq!(y_slice[9], 3.0);
    assert_eq!(y_slice[10], 4.0);

    let grad_out = Tensor::ones_on(vec![4, 4], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_squeeze_unsqueeze_autograd() {
    let backend = MoiraiBackend::new();

    // Test Unsqueeze: [2, 3] -> [2, 1, 3]
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::unsqueeze(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1, 3]);

    let grad_out = Tensor::from_slice_on(
        vec![2, 1, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(gx.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Test Squeeze (specific axis): [2, 1, 3] -> [2, 3]
    let x2_val = Tensor::from_slice_on(vec![2, 1, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x2 = Var::new(x2_val, true);

    let y2 = coeus_autograd::squeeze(&x2, Some(1));
    assert_eq!(y2.tensor.shape(), &[2, 3]);

    let grad_out2 = Tensor::from_slice_on(
        vec![2, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y2.backward_with_seed(grad_out2);

    let gx2 = x2.grad().unwrap();
    assert_eq!(gx2.shape(), &[2, 1, 3]);
    assert_eq!(gx2.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Test Squeeze (all axes): [1, 2, 1, 3] -> [2, 3]
    let x3_val = Tensor::from_slice_on(
        vec![1, 2, 1, 3],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &backend,
    );
    let x3 = Var::new(x3_val, true);

    let y3 = coeus_autograd::squeeze(&x3, None);
    assert_eq!(y3.tensor.shape(), &[2, 3]);

    let grad_out3 = Tensor::from_slice_on(
        vec![2, 3],
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &backend,
    );
    y3.backward_with_seed(grad_out3);

    let gx3 = x3.grad().unwrap();
    assert_eq!(gx3.shape(), &[1, 2, 1, 3]);
    assert_eq!(gx3.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
}

/// Verify contiguous() backward is identity — gradient flows through unchanged.
#[test]
fn test_contiguous_backward_is_identity() {
    let backend = MoiraiBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(Tensor::from_slice_on(vec![2, 3], &data, &backend), true);
    // Permute to create a non-contiguous view, then force contiguous.
    let y = coeus_autograd::contiguous(&coeus_autograd::permute(&x, &[1, 0]));
    assert_eq!(y.tensor.shape(), &[3, 2]);
    // sum(contiguous(permute(x))).backward() — grad should be all-ones (same as sum backward).
    coeus_autograd::sum(&y).backward();
    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    // Every element contributed once to the sum, so all grads = 1.
    for &v in gx.as_slice() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "contiguous bwd grad should be 1.0, got {v}"
        );
    }
}
