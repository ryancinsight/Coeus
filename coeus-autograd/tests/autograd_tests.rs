use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use coeus_autograd::{Var, exp, log, sum_axis, mean_axis, sparse_matmul, embedding,
                     max_axis, min_axis, log_sum_exp, cumsum, scalar_sub, scalar_div};



#[test]
fn test_exp_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[0.0f32, 1.0f32, 2.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = exp(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-5);
    assert!((y_slice[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((y_slice[2] - 7.389056).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 5.4365636).abs() < 1e-5);
    assert!((gx_slice[2] - 22.167168).abs() < 1e-5);
}

#[test]
fn test_log_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 4.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = log(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 0.0).abs() < 1e-5);
    assert!((y_slice[1] - std::f32::consts::LN_2).abs() < 1e-5);
    assert!((y_slice[2] - 2.0f32 * std::f32::consts::LN_2).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 1.0).abs() < 1e-5);
    assert!((gx_slice[2] - 0.75).abs() < 1e-5);
}

#[test]
fn test_sum_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = sum_axis(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1]);
    let y_slice = y.tensor.as_slice();
    assert_eq!(y_slice[0], 6.0);
    assert_eq!(y_slice[1], 15.0);

    let grad_out = Tensor::from_slice_on(vec![2, 1], &[2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_mean_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = mean_axis(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1]);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 2.0).abs() < 1e-5);
    assert!((y_slice[1] - 5.0).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![2, 1], &[3.0f32, 6.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_sparse_matmul_backward() {
    let backend = MoiraiBackend::new();
    
    // A [3, 4] (sparse)
    let a_data = vec![
        1.0f32, 0.0, 2.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 4.0, 0.0, 5.0,
    ];
    let a_dense = Tensor::from_slice_on(vec![3, 4], &a_data, &backend);
    let csr = coeus_ops::dense_to_csr(&a_dense, &backend);

    // B [4, 2] (dense)
    let b_data = vec![
        1.0f32, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ];
    let b_dense = Tensor::from_slice_on(vec![4, 2], &b_data, &backend);

    // Tracked dense baseline
    let a_var = Var::new(a_dense, true);
    let b_var = Var::new(b_dense.clone(), true);
    let c_dense_var = coeus_autograd::matmul(&a_var, &b_var);

    // Seed output gradient
    let grad_out_data = vec![
        1.0f32, -1.0,
        2.0, -2.0,
        3.0, -3.0,
    ];
    let grad_out = Tensor::from_slice_on(vec![3, 2], &grad_out_data, &backend);
    c_dense_var.backward_with_seed(grad_out.clone());

    let expected_grad_a = a_var.grad().unwrap();
    let expected_grad_b = b_var.grad().unwrap();

    // Tracked sparse MatMul
    let csr_values = csr.values().clone();
    let csr_col_indices = csr.col_indices().clone();
    let csr_row_offsets = csr.row_offsets().clone();

    let a_values_var = Var::new(csr_values, true);
    let b_var_sparse = Var::new(b_dense, true);

    let c_sparse_var = sparse_matmul(
        &a_values_var,
        &csr_col_indices,
        &csr_row_offsets,
        coeus_core::Shape::from(vec![3, 4]),
        &b_var_sparse,
    );

    // Verify forward parity
    assert_eq!(c_sparse_var.tensor.as_slice(), c_dense_var.tensor.as_slice());

    c_sparse_var.backward_with_seed(grad_out);

    let grad_a_vals = a_values_var.grad().unwrap();
    let grad_b_sparse = b_var_sparse.grad().unwrap();

    // Verify backward values parity
    let expected_grad_a_slice = expected_grad_a.as_slice();
    let grad_a_vals_slice = grad_a_vals.as_slice();
    let col_slice = csr_col_indices.as_slice();
    let row_slice = csr_row_offsets.as_slice();

    let mut val_idx = 0;
    for r in 0..3 {
        let start = row_slice[r] as usize;
        let end = row_slice[r + 1] as usize;
        for i in start..end {
            let c = col_slice[i] as usize;
            let dense_idx = r * 4 + c;
            let expected = expected_grad_a_slice[dense_idx];
            let actual = grad_a_vals_slice[val_idx];
            assert!((actual - expected).abs() < 1e-5, "Mismatch at r={}, c={}: actual={}, expected={}", r, c, actual, expected);
            val_idx += 1;
        }
    }

    // Verify backward dense parity
    let grad_b_sparse_slice = grad_b_sparse.as_slice();
    let expected_grad_b_slice = expected_grad_b.as_slice();
    for i in 0..grad_b_sparse_slice.len() {
        assert!((grad_b_sparse_slice[i] - expected_grad_b_slice[i]).abs() < 1e-5);
    }
}

#[test]
fn test_embedding_autograd() {
    let backend = MoiraiBackend::new();
    
    // Weight matrix of shape [3, 2]
    // 3 embeddings, each of dimension 2
    let w_data = vec![
        1.0f32, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ];
    let w_tensor = Tensor::from_slice_on(vec![3, 2], &w_data, &backend);
    let weight = Var::new(w_tensor, true);
    
    // Indices of shape [2, 2]
    let idx_data = vec![0i32, 2, 1, 0];
    let indices = Tensor::from_slice_on(vec![2, 2], &idx_data, &backend);
    
    // Perform embedding lookup
    let y = embedding(&weight, &indices);
    assert_eq!(y.tensor.shape(), &[2, 2, 2]);
    let y_slice = y.tensor.as_slice();
    // Expected output:
    // row 0: idx 0 -> [1.0, 2.0]
    // row 1: idx 2 -> [5.0, 6.0]
    // row 2: idx 1 -> [3.0, 4.0]
    // row 3: idx 0 -> [1.0, 2.0]
    assert_eq!(y_slice, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    
    // Output gradient of shape [2, 2, 2]
    let grad_out_data = vec![
        1.0f32, 1.0,
        2.0, 2.0,
        3.0, 3.0,
        4.0, 4.0,
    ];
    let grad_out = Tensor::from_slice_on(vec![2, 2, 2], &grad_out_data, &backend);
    y.backward_with_seed(grad_out);
    
    // Expected weight gradient of shape [3, 2]:
    // idx 0 selected twice (row 0, row 3): grad_out[0] + grad_out[3] = [1.0+4.0, 1.0+4.0] = [5.0, 5.0]
    // idx 1 selected once (row 2): grad_out[2] = [3.0, 3.0]
    // idx 2 selected once (row 1): grad_out[1] = [2.0, 2.0]
    let gw = weight.grad().unwrap();
    let gw_slice = gw.as_slice();
    assert_eq!(gw_slice, &[5.0, 5.0, 3.0, 3.0, 2.0, 2.0]);
}

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
    
    let grad_out = Tensor::from_slice_on(vec![2, 1, 3], &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], &backend);
    y.backward_with_seed(grad_out);
    
    let gx = x.grad().unwrap();
    assert_eq!(gx.shape(), &[2, 3]);
    assert_eq!(gx.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    
    // Test Squeeze (specific axis): [2, 1, 3] -> [2, 3]
    let x2_val = Tensor::from_slice_on(vec![2, 1, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x2 = Var::new(x2_val, true);
    
    let y2 = coeus_autograd::squeeze(&x2, Some(1));
    assert_eq!(y2.tensor.shape(), &[2, 3]);
    
    let grad_out2 = Tensor::from_slice_on(vec![2, 3], &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], &backend);
    y2.backward_with_seed(grad_out2);
    
    let gx2 = x2.grad().unwrap();
    assert_eq!(gx2.shape(), &[2, 1, 3]);
    assert_eq!(gx2.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    // Test Squeeze (all axes): [1, 2, 1, 3] -> [2, 3]
    let x3_val = Tensor::from_slice_on(vec![1, 2, 1, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &backend);
    let x3 = Var::new(x3_val, true);
    
    let y3 = coeus_autograd::squeeze(&x3, None);
    assert_eq!(y3.tensor.shape(), &[2, 3]);
    
    let grad_out3 = Tensor::from_slice_on(vec![2, 3], &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], &backend);
    y3.backward_with_seed(grad_out3);
    
    let gx3 = x3.grad().unwrap();
    assert_eq!(gx3.shape(), &[1, 2, 1, 3]);
    assert_eq!(gx3.as_slice(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
}

// ── Unary Math Op Tests ────────────────────────────────────────────────────

#[test]
fn test_neg_autograd() {
    // d/dx [−x] = −1, so grad_in = −grad_out.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[1.0f64, -2.0, 3.0, 0.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::neg(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - (-1.0)).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - (-3.0)).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // grad_in = −grad_out
    assert!((gx_s[0] - (-1.0)).abs() < 1e-10);
    assert!((gx_s[1] - (-2.0)).abs() < 1e-10);
    assert!((gx_s[2] - (-3.0)).abs() < 1e-10);
    assert!((gx_s[3] - (-4.0)).abs() < 1e-10);
}

#[test]
fn test_abs_autograd() {
    // d/dx |x| = sign(x): +1 for x>0, -1 for x<0, 0 for x=0.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[3.0f64, -2.0, 0.5], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::abs(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - 0.5).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 1.0).abs() < 1e-10,  "expected +1 for x>0, got {}", gx_s[0]);
    assert!((gx_s[1] - (-1.0)).abs() < 1e-10, "expected -1 for x<0, got {}", gx_s[1]);
    assert!((gx_s[2] - 1.0).abs() < 1e-10,  "expected +1 for x>0, got {}", gx_s[2]);
}

#[test]
fn test_sqrt_autograd() {
    // d/dx √x = 1/(2√x); use y = √x stored in forward to avoid redundant sqrt.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[4.0f64, 9.0, 16.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::sqrt(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 2.0).abs() < 1e-9);
    assert!((y_slice[1] - 3.0).abs() < 1e-9);
    assert!((y_slice[2] - 4.0).abs() < 1e-9);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // 1/(2·√4) = 0.25, 1/(2·3) ≈ 0.1667, 1/(2·4) = 0.125
    assert!((gx_s[0] - 0.25).abs() < 1e-9,   "sqrt backward at 4: {}", gx_s[0]);
    assert!((gx_s[1] - (1.0/6.0)).abs() < 1e-9, "sqrt backward at 9: {}", gx_s[1]);
    assert!((gx_s[2] - 0.125).abs() < 1e-9,  "sqrt backward at 16: {}", gx_s[2]);
}

#[test]
fn test_pow_autograd() {
    // d/dx x^3 = 3·x^2; verify at x=2: d/dx = 12.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::pow(&x, 3.0);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-8);
    assert!((y_slice[1] - 8.0).abs() < 1e-8);
    assert!((y_slice[2] - 27.0).abs() < 1e-8);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // 3·1^2=3, 3·4=12, 3·9=27
    assert!((gx_s[0] - 3.0).abs() < 1e-6,  "pow(1,3) grad: {}", gx_s[0]);
    assert!((gx_s[1] - 12.0).abs() < 1e-6, "pow(2,3) grad: {}", gx_s[1]);
    assert!((gx_s[2] - 27.0).abs() < 1e-6, "pow(3,3) grad: {}", gx_s[2]);
}

#[test]
fn test_clamp_autograd() {
    // clamp(x, 0, 2): grad=1 inside [0,2], 0 outside.
    //  x = [-1, 0.5, 1.5, 2.5]
    //  clamped = [0, 0.5, 1.5, 2]
    //  d/dx = [0, 1, 1, 0]
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[-1.0f64, 0.5, 1.5, 2.5], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::clamp(&x, 0.0f64, 2.0f64);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 0.0).abs() < 1e-10, "clamp(-1) = {}", y_slice[0]);
    assert!((y_slice[1] - 0.5).abs() < 1e-10, "clamp(0.5) = {}", y_slice[1]);
    assert!((y_slice[2] - 1.5).abs() < 1e-10, "clamp(1.5) = {}", y_slice[2]);
    assert!((y_slice[3] - 2.0).abs() < 1e-10, "clamp(2.5) = {}", y_slice[3]);

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 1.0, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // At x=-1 (< 0): 0; at x=0.5, 1.5 (inside): 1; at x=2.5 (> 2): 0
    assert!((gx_s[0] - 0.0).abs() < 1e-10, "clamp grad at -1: {}", gx_s[0]);
    assert!((gx_s[1] - 1.0).abs() < 1e-10, "clamp grad at 0.5: {}", gx_s[1]);
    assert!((gx_s[2] - 1.0).abs() < 1e-10, "clamp grad at 1.5: {}", gx_s[2]);
    assert!((gx_s[3] - 0.0).abs() < 1e-10, "clamp grad at 2.5: {}", gx_s[3]);
}

#[test]
fn test_scalar_mul_autograd() {
    // scalar_mul(x, 3): forward = 3x, grad = 3 * grad_out.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::scalar_mul(&x, 3.0f64);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 6.0).abs() < 1e-10);
    assert!((y_slice[2] - 9.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // grad_in = 3 * grad_out
    assert!((gx_s[0] - 3.0).abs() < 1e-10);
    assert!((gx_s[1] - 6.0).abs() < 1e-10);
    assert!((gx_s[2] - 9.0).abs() < 1e-10);
}

#[test]
fn test_var_ops_overloads() {
    // Verify that std::ops trait impls produce the same result as the free functions
    // and correctly track gradients.
    let backend = MoiraiBackend::new();
    let a_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    let b_val = Tensor::from_slice_on(vec![3], &[4.0f64, 5.0, 6.0], &backend);

    let a = Var::new(a_val, true);
    let b = Var::new(b_val, true);

    // Test + operator: a + b
    let sum = {
        use coeus_autograd::add;
        add(&a, &b)
    };
    let sum_op = &a + &b;
    assert_eq!(sum.tensor.as_slice(), sum_op.tensor.as_slice());

    // Test - operator: a - b
    let diff_op = &a - &b;
    let diff_s = diff_op.tensor.as_slice();
    assert!((diff_s[0] - (-3.0)).abs() < 1e-10);
    assert!((diff_s[1] - (-3.0)).abs() < 1e-10);
    assert!((diff_s[2] - (-3.0)).abs() < 1e-10);

    // Test * operator: a * b (element-wise)
    let prod_op = &a * &b;
    let prod_s = prod_op.tensor.as_slice();
    assert!((prod_s[0] - 4.0).abs() < 1e-10);
    assert!((prod_s[1] - 10.0).abs() < 1e-10);
    assert!((prod_s[2] - 18.0).abs() < 1e-10);

    // Test unary Neg: -&a
    let neg_op = -&a;
    let neg_s = neg_op.tensor.as_slice();
    assert!((neg_s[0] - (-1.0)).abs() < 1e-10);
    assert!((neg_s[1] - (-2.0)).abs() < 1e-10);

    // Test scalar Mul: &a * 5.0
    let scaled = {
        use coeus_autograd::scalar_mul;
        scalar_mul(&a, 5.0f64)
    };
    let scaled_op = &a * 5.0f64;
    assert_eq!(scaled.tensor.as_slice(), scaled_op.tensor.as_slice());

    // Gradient check: (a * b).sum().backward() — grad_a = b, grad_b = a
    let a2 = Var::new(Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend), true);
    let b2 = Var::new(Tensor::from_slice_on(vec![3], &[4.0f64, 5.0, 6.0], &backend), true);
    let prod2 = &a2 * &b2;
    let loss2 = coeus_autograd::sum(&prod2);
    loss2.backward();
    let ga2 = a2.grad().unwrap();
    let gb2 = b2.grad().unwrap();
    // d/da sum(a*b) = b
    assert!((ga2.as_slice()[0] - 4.0).abs() < 1e-10);
    assert!((ga2.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((ga2.as_slice()[2] - 6.0).abs() < 1e-10);
    // d/db sum(a*b) = a
    assert!((gb2.as_slice()[0] - 1.0).abs() < 1e-10);
    assert!((gb2.as_slice()[1] - 2.0).abs() < 1e-10);
    assert!((gb2.as_slice()[2] - 3.0).abs() < 1e-10);
}

// ── New Reduction Op Tests ─────────────────────────────────────────────────

#[test]
fn test_max_axis_autograd() {
    // x = [[1, 3, 2], [4, 1, 5]]  max along axis=1 → [[3], [5]]
    // Backward: indicator at argmax position, uniform at ties.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(
        vec![2, 3],
        &[1.0f64, 3.0, 2.0, 4.0, 1.0, 5.0],
        &backend,
    );
    let x = Var::new(x_val, true);

    let y = max_axis(&x, 1);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10, "max row 0: {}", y_slice[0]);
    assert!((y_slice[1] - 5.0).abs() < 1e-10, "max row 1: {}", y_slice[1]);
    assert_eq!(y.tensor.shape(), &[2, 1]);

    // seed: ones → each row gets grad 1.0 distributed to argmax position.
    let seed = Tensor::from_slice_on(vec![2, 1], &[1.0f64, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // Row 0: argmax at index 1 (value 3), others 0.
    assert!((gx_s[0] - 0.0).abs() < 1e-10, "[0,0]: {}", gx_s[0]);
    assert!((gx_s[1] - 1.0).abs() < 1e-10, "[0,1]: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "[0,2]: {}", gx_s[2]);
    // Row 1: argmax at index 2 (value 5), others 0.
    assert!((gx_s[3] - 0.0).abs() < 1e-10, "[1,0]: {}", gx_s[3]);
    assert!((gx_s[4] - 0.0).abs() < 1e-10, "[1,1]: {}", gx_s[4]);
    assert!((gx_s[5] - 1.0).abs() < 1e-10, "[1,2]: {}", gx_s[5]);
}

#[test]
fn test_max_axis_tie_normalisation() {
    // x = [2, 2, 1] — two maxima tied at value 2.
    // Backward: gradient 1.0 split equally → 0.5 at each of positions 0 and 1.
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[2.0f64, 2.0, 1.0], &backend);
    let x = Var::new(x_val, true);
    let y = max_axis(&x, 0);
    let seed = Tensor::from_slice_on(vec![1], &[1.0f64], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 0.5).abs() < 1e-8, "tie pos 0: {}", gx_s[0]);
    assert!((gx_s[1] - 0.5).abs() < 1e-8, "tie pos 1: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "non-max: {}", gx_s[2]);
}

#[test]
fn test_min_axis_autograd() {
    // x = [[1, 3, 2], [4, 1, 5]]  min along axis=1 → [[1], [1]]
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(
        vec![2, 3],
        &[1.0f64, 3.0, 2.0, 4.0, 1.0, 5.0],
        &backend,
    );
    let x = Var::new(x_val, true);
    let y = min_axis(&x, 1);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-10);
    assert!((y_slice[1] - 1.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![2, 1], &[1.0f64, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    // Row 0: argmin at index 0 (value 1).
    assert!((gx_s[0] - 1.0).abs() < 1e-10, "[0,0]: {}", gx_s[0]);
    assert!((gx_s[1] - 0.0).abs() < 1e-10, "[0,1]: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "[0,2]: {}", gx_s[2]);
    // Row 1: argmin at index 1 (value 1).
    assert!((gx_s[3] - 0.0).abs() < 1e-10, "[1,0]: {}", gx_s[3]);
    assert!((gx_s[4] - 1.0).abs() < 1e-10, "[1,1]: {}", gx_s[4]);
    assert!((gx_s[5] - 0.0).abs() < 1e-10, "[1,2]: {}", gx_s[5]);
}

#[test]
fn test_log_sum_exp_autograd() {
    // log_sum_exp(x, axis=0) — gradient should equal softmax(x).
    // x = [1.0, 2.0, 3.0]
    // softmax = [e^1, e^2, e^3] / (e^1+e^2+e^3)
    let backend = MoiraiBackend::new();
    let vals = [1.0f64, 2.0, 3.0];
    let x_val = Tensor::from_slice_on(vec![3], &vals, &backend);
    let x = Var::new(x_val, true);

    let lse = log_sum_exp(&x, 0);  // shape [1]
    let lse_val = lse.tensor.as_slice()[0];
    // Expected: log(e + e^2 + e^3).
    let e1 = 1.0f64.exp();
    let e2 = 2.0f64.exp();
    let e3 = 3.0f64.exp();
    let expected_lse = (e1 + e2 + e3).ln();
    assert!((lse_val - expected_lse).abs() < 1e-9, "lse value: {} vs {}", lse_val, expected_lse);

    // Backward with seed 1.0 → grad_x = softmax(x).
    let seed = Tensor::from_slice_on(vec![1], &[1.0f64], &backend);
    lse.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    let sum_exp = e1 + e2 + e3;
    let sm0 = e1 / sum_exp;
    let sm1 = e2 / sum_exp;
    let sm2 = e3 / sum_exp;
    assert!((gx_s[0] - sm0).abs() < 1e-9, "lse grad[0]: {} vs {}", gx_s[0], sm0);
    assert!((gx_s[1] - sm1).abs() < 1e-9, "lse grad[1]: {} vs {}", gx_s[1], sm1);
    assert!((gx_s[2] - sm2).abs() < 1e-9, "lse grad[2]: {} vs {}", gx_s[2], sm2);
    // Also verify softmax probabilities sum to 1.
    let grad_sum = gx_s[0] + gx_s[1] + gx_s[2];
    assert!((grad_sum - 1.0).abs() < 1e-9, "softmax sums to {}", grad_sum);
}

#[test]
fn test_cumsum_autograd() {
    // cumsum([a, b, c], axis=0) = [a, a+b, a+b+c]
    // Backward: suffix_sum(grad_out)
    //   d/da = grad[0] + grad[1] + grad[2]
    //   d/db = grad[1] + grad[2]
    //   d/dc = grad[2]
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    let x = Var::new(x_val, true);

    let y = cumsum(&x, 0);
    let y_s = y.tensor.as_slice();
    assert!((y_s[0] - 1.0).abs() < 1e-10);
    assert!((y_s[1] - 3.0).abs() < 1e-10);
    assert!((y_s[2] - 6.0).abs() < 1e-10);
    assert!((y_s[3] - 10.0).abs() < 1e-10);

    // Seed: [1, 2, 3, 4]
    // Expected grad_in:
    //   d/d[0] = 1+2+3+4 = 10
    //   d/d[1] = 2+3+4   = 9
    //   d/d[2] = 3+4     = 7
    //   d/d[3] = 4       = 4
        let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 10.0).abs() < 1e-10, "cumsum grad[0]: {}", gx_s[0]);
    assert!((gx_s[1] -  9.0).abs() < 1e-10, "cumsum grad[1]: {}", gx_s[1]);
    assert!((gx_s[2] -  7.0).abs() < 1e-10, "cumsum grad[2]: {}", gx_s[2]);
    assert!((gx_s[3] -  4.0).abs() < 1e-10, "cumsum grad[3]: {}", gx_s[3]);
}

#[test]
fn test_scalar_sub_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[5.0f64, 8.0, 12.0], &backend);
    let x = Var::new(x_val, true);

    // Test free function and operator overload
    let y = scalar_sub(&x, 3.0);
    let z = &x - 3.0;

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    // Test backward pass for subtraction
    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed.clone());
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_scalar_div_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[6.0f64, 12.0, 18.0], &backend);
    let x = Var::new(x_val, true);

    // Test free function and operator overload
    let y = scalar_div(&x, 3.0);
    let z = &x / 3.0;

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    // Test backward pass: d/dx (x / s) = 1 / s
    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0 / 3.0).abs() < 1e-10);
}




