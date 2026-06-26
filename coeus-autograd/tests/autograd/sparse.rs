use coeus_autograd::{sparse_matmul, sparse_matmul_coo, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
#[allow(clippy::needless_range_loop)]
fn test_sparse_matmul_backward() {
    let backend = MoiraiBackend::new();

    // A [3, 4] (sparse)
    let a_data = vec![
        1.0f32, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0,
    ];
    let a_dense = Tensor::from_slice_on(vec![3, 4], &a_data, &backend);
    let csr = coeus_ops::dense_to_csr(&a_dense, &backend);

    // B [4, 2] (dense)
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_dense = Tensor::from_slice_on(vec![4, 2], &b_data, &backend);

    // Tracked dense baseline
    let a_var = Var::new(a_dense, true);
    let b_var = Var::new(b_dense.clone(), true);
    let c_dense_var = coeus_autograd::matmul(&a_var, &b_var);

    // Seed output gradient
    let grad_out_data = vec![1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0];
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
    assert_eq!(
        c_sparse_var.tensor.as_slice(),
        c_dense_var.tensor.as_slice()
    );

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
            assert!(
                (actual - expected).abs() < 1e-5,
                "Mismatch at r={}, c={}: actual={}, expected={}",
                r,
                c,
                actual,
                expected
            );
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
#[allow(clippy::needless_range_loop)]
fn test_sparse_coo_matmul_backward() {
    let backend = MoiraiBackend::new();

    // A [3, 4] (sparse COO)
    let a_data = vec![
        1.0f32, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0,
    ];
    let a_dense = Tensor::from_slice_on(vec![3, 4], &a_data, &backend);
    let coo = coeus_ops::dense_to_coo(&a_dense, &backend);

    // B [4, 2] (dense)
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_dense = Tensor::from_slice_on(vec![4, 2], &b_data, &backend);

    // Tracked dense baseline
    let a_var = Var::new(a_dense, true);
    let b_var = Var::new(b_dense.clone(), true);
    let c_dense_var = coeus_autograd::matmul(&a_var, &b_var);

    // Seed output gradient
    let grad_out_data = vec![1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0];
    let grad_out = Tensor::from_slice_on(vec![3, 2], &grad_out_data, &backend);
    c_dense_var.backward_with_seed(grad_out.clone());

    let expected_grad_a = a_var.grad().unwrap();
    let expected_grad_b = b_var.grad().unwrap();

    // Tracked sparse COO MatMul
    let a_values_var = Var::new(coo.values().clone(), true);
    let b_var_sparse = Var::new(b_dense, true);

    let c_sparse_var = sparse_matmul_coo(
        &a_values_var,
        coo.indices(),
        coo.shape().clone(),
        &b_var_sparse,
    );

    // Verify forward parity
    assert_eq!(
        c_sparse_var.tensor.as_slice(),
        c_dense_var.tensor.as_slice()
    );

    c_sparse_var.backward_with_seed(grad_out);

    let grad_a_vals = a_values_var.grad().unwrap();
    let grad_b_sparse = b_var_sparse.grad().unwrap();

    // Verify backward values parity in COO order
    let expected_grad_a_slice = expected_grad_a.as_slice();
    let grad_a_vals_slice = grad_a_vals.as_slice();
    let idx_slice = coo.indices().as_slice();
    let nnz = coo.nnz();

    for i in 0..nnz {
        let r = idx_slice[i] as usize;
        let c = idx_slice[nnz + i] as usize;
        let dense_idx = r * 4 + c;
        let expected = expected_grad_a_slice[dense_idx];
        let actual = grad_a_vals_slice[i];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Mismatch at COO value index {} (r={}, c={}): actual={}, expected={}",
            i,
            r,
            c,
            actual,
            expected
        );
    }

    // Verify backward dense parity
    let grad_b_sparse_slice = grad_b_sparse.as_slice();
    let expected_grad_b_slice = expected_grad_b.as_slice();
    for i in 0..grad_b_sparse_slice.len() {
        assert!((grad_b_sparse_slice[i] - expected_grad_b_slice[i]).abs() < 1e-5);
    }
}
