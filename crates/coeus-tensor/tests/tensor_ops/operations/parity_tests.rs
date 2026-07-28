use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::{Tensor, Transpose};

/// Self-contained row-major matmul reference: `c[m,n] = a[m,k] · b[k,n]`.
/// Independent of both coeus's implementation and any external array library.
fn matmul_ref(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn test_elementwise_add_parity() {
    let backend = SequentialBackend::new();
    let shape = vec![3, 4];

    // Coeus tensors
    let mut a = Tensor::<f32, SequentialBackend>::zeros(shape.clone()).expect("construct tensor");
    let mut b = Tensor::<f32, SequentialBackend>::zeros(shape.clone()).expect("construct tensor");
    {
        let a_slice = a.as_mut_slice().expect("mutable tensor slice");
        let b_slice = b.as_mut_slice().expect("mutable tensor slice");
        for i in 0..12 {
            a_slice[i] = i as f32;
            b_slice[i] = (i * 2) as f32;
        }
    }

    // Run Coeus addition
    let c = coeus_ops::add(&a, &b, &backend).expect("run addition");

    // Self-contained reference: a[i] = i, b[i] = 2i  =>  c[i] = 3i.
    for (i, &got) in c.as_slice().iter().enumerate() {
        let expected = (i as f32) + (i * 2) as f32;
        assert_eq!(got, expected);
    }
}

#[test]
fn test_relu_parity() {
    let backend = SequentialBackend::new();
    let shape = vec![2, 3];
    let mut a = Tensor::<f32, SequentialBackend>::zeros(shape).expect("construct tensor");
    {
        let slice = a.as_mut_slice().expect("mutable tensor slice");
        slice[0] = -1.5;
        slice[1] = 2.0;
        slice[2] = -0.5;
        slice[3] = 0.0;
        slice[4] = 1.0;
        slice[5] = -3.0;
    }

    let b = coeus_ops::relu(&a, &backend).expect("run ReLU");
    let slice = b.as_slice();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 2.0);
    assert_eq!(slice[2], 0.0);
    assert_eq!(slice[3], 0.0);
    assert_eq!(slice[4], 1.0);
    assert_eq!(slice[5], 0.0);
}

#[test]
fn test_non_contiguous_matmul_parity() {
    let backend = SequentialBackend::new();

    // 2x3 matrix (row-major)
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data).expect("construct tensor");

    // 3x2 matrix (row-major)
    let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 2], &b_data).expect("construct tensor");

    // Standard matmul: [2, 3] x [3, 2] -> [2, 2]
    let c = coeus_ops::matmul(&a, &b, &backend).expect("run matrix multiplication");

    // Self-contained reference (row-major triple loop).
    let ref_c = matmul_ref(&a_data, 2, 3, &b_data, 2);
    assert_eq!(c.as_slice(), ref_c.as_slice());

    // Transposed matmul test (non-contiguous)
    // a_t shape: [3, 2], strides: [1, 3]
    let a_t = a.transpose();
    // matmul(a_t, a) -> [3, 2] x [2, 3] -> [3, 3]
    let c_t = coeus_ops::matmul(&a_t, &a, &backend).expect("run transposed matrix multiplication");

    // Reference: build a_t in row-major ([3,2], a_t[i][j] = a[j][i]) then matmul.
    let mut a_t_data = vec![0.0f32; 6];
    for i in 0..3 {
        for j in 0..2 {
            a_t_data[i * 2 + j] = a_data[j * 3 + i];
        }
    }
    let ref_c_t = matmul_ref(&a_t_data, 3, 2, &a_data, 3);

    let shape_c = c_t.shape();
    assert_eq!(shape_c, &[3, 3]);
    for r in 0..3 {
        for col in 0..3 {
            assert_eq!(c_t.get(&[r, col]), ref_c_t[r * 3 + col]);
        }
    }
}

#[test]
fn test_reductions_parity() {
    let backend = SequentialBackend::new();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data).expect("construct tensor");

    // Test sum all
    let total = coeus_ops::sum(&a, &backend).expect("valid sum");
    assert_eq!(total, 21.0);

    // Test sum along axis 0 (reduces row dimension) -> [1, 3]
    let sum_r = coeus_ops::sum_axis(&a, 0, &backend).expect("valid sum axis");
    assert_eq!(sum_r.shape(), &[1, 3]);
    assert_eq!(sum_r.as_slice(), &[5.0, 7.0, 9.0]);

    // Test sum along axis 1 (reduces col dimension) -> [2, 1]
    let sum_c = coeus_ops::sum_axis(&a, 1, &backend).expect("valid sum axis");
    assert_eq!(sum_c.shape(), &[2, 1]);
    assert_eq!(sum_c.as_slice(), &[6.0, 15.0]);

    // Test mean along axis 1 -> [2, 1]
    let mean_c = coeus_ops::mean_axis(&a, 1, &backend).expect("valid mean axis");
    assert_eq!(mean_c.shape(), &[2, 1]);
    assert_eq!(mean_c.as_slice(), &[2.0, 5.0]);
}

#[test]
fn test_sparse_spmv_spmm_parity() {
    let backend = SequentialBackend::new();

    // Dense 3x3 matrix with some zeros
    // A = [ 0.0  2.0  0.0 ]
    //     [ 1.0  0.0  3.0 ]
    //     [ 0.0  4.0  5.0 ]
    let dense_data = vec![0.0f32, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 5.0];
    let a = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 3], &dense_data).expect("construct tensor");

    // Convert to CSR
    let csr = coeus_ops::dense_to_csr(&a, &backend).expect("convert dense tensor to CSR");
    assert_eq!(csr.nnz(), 5);

    // 1. SpMV parity test: y = A x
    let x_data = vec![2.0f32, 1.0, 3.0];
    let x = Tensor::<f32, SequentialBackend>::from_vec(x_data.clone()).expect("construct tensor");

    let y_sparse = coeus_ops::spmv(&csr, &x, &backend).expect("run sparse matrix-vector multiplication");

    // Dense equivalent matmul: requires 2D column vector
    let x_dense = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 1], &x_data).expect("construct tensor");
    let y_dense = coeus_ops::matmul(&a, &x_dense, &backend).expect("run dense matrix-vector multiplication");

    assert_eq!(y_sparse.as_slice(), y_dense.as_slice());

    // 2. SpMM parity test: C = A B
    let b_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 2], &b_data).expect("construct tensor");

    let c_sparse = coeus_ops::spmm(&csr, &b, &backend).expect("run sparse matrix multiplication");
    let c_dense = coeus_ops::matmul(&a, &b, &backend).expect("run dense matrix multiplication");

    assert_eq!(c_sparse.as_slice(), c_dense.as_slice());
}

#[test]
fn test_to_backend_transfers() {
    let seq_backend = SequentialBackend::new();
    let moirai_backend = MoiraiBackend::new();

    // 1. Contiguous tensor transfer
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &data).expect("construct tensor");
    let a_moirai = a_seq
        .to_backend_on(&seq_backend, &moirai_backend)
        .expect("transfer tensor to Moirai");

    assert_eq!(a_moirai.shape(), &[2, 3]);
    assert!(a_moirai.is_contiguous());
    assert_eq!(a_moirai.as_slice(), &data);

    // Transfer back
    let a_seq_back = a_moirai
        .to_backend_on(&moirai_backend, &seq_backend)
        .expect("transfer tensor to sequential backend");
    assert_eq!(a_seq_back.as_slice(), &data);

    // 2. Non-contiguous (transposed) tensor transfer
    let a_t_seq = a_seq.transpose(); // shape [3, 2]
    assert!(!a_t_seq.is_contiguous());
    let a_t_moirai = a_t_seq
        .to_backend_on(&seq_backend, &moirai_backend)
        .expect("transfer transposed tensor to Moirai");

    // The transferred tensor is contiguous on the new backend
    assert_eq!(a_t_moirai.shape(), &[3, 2]);
    assert!(a_t_moirai.is_contiguous());

    // Elements should match transposed layout: [1, 4, 2, 5, 3, 6]
    let expected_t = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(a_t_moirai.as_slice(), &expected_t);
}

#[test]
fn test_sliced_tensor_offset_ops() {
    let backend = MoiraiBackend::new();

    // Contiguous tensor [3, 4]
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Tensor::<f32, MoiraiBackend>::from_slice(vec![3, 4], &data).expect("construct tensor");

    // Slice [1..3, 1..3] -> shape [2, 2], starts at index 5 (which is row 1, col 1)
    let slice_a = a.slice(&[(1, 3), (1, 3)]);
    assert_eq!(slice_a.shape(), &[2, 2]);
    assert_eq!(slice_a.layout().offset(), 5);

    // 1D contiguous sliced view
    let a_1d = Tensor::<f32, MoiraiBackend>::from_slice(vec![12], &data).expect("construct tensor");
    let slice_a_1d = a_1d.slice(&[(4, 8)]);
    assert_eq!(slice_a_1d.shape(), &[4]);
    assert_eq!(slice_a_1d.layout().offset(), 4);
    assert!(slice_a_1d.is_contiguous());

    let b_1d = Tensor::<f32, MoiraiBackend>::from_slice(vec![4], &[2.0, 3.0, 4.0, 5.0]).expect("construct tensor");

    // Test binary addition: contiguous fast path with non-zero offset on input A
    let sum_1d = coeus_ops::add(&slice_a_1d, &b_1d, &backend).expect("run sliced addition");
    assert_eq!(sum_1d.as_slice(), &[7.0, 9.0, 11.0, 13.0]);

    // Test in-place assign on a contiguous sliced tensor: contiguous fast path with non-zero offset on target
    let target_1d = Tensor::<f32, MoiraiBackend>::zeros_on(vec![8], &backend).expect("construct tensor");
    let mut slice_target = target_1d.slice(&[(2, 6)]); // shape [4], offset 2
    assert!(slice_target.is_contiguous());

    coeus_ops::add_assign(&mut slice_target, &b_1d, &backend).expect("same-shape sliced addition");
    // Value semantics / COW: mutating slice_target triggers COW and detaches from target_1d
    assert_eq!(slice_target.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    assert_eq!(target_1d.as_slice(), &[0.0; 8]);

    // Test strided broadcasting path with offsets
    let b_2d = Tensor::<f32, MoiraiBackend>::from_slice(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("construct tensor");
    let sum_2d = coeus_ops::add(&slice_a, &b_2d, &backend).expect("run broadcast addition");
    assert_eq!(sum_2d.as_slice(), &[7.0, 9.0, 13.0, 15.0]);

    // Test unary operations on sliced contiguous/non-contiguous
    let relu_1d = coeus_ops::relu(&slice_a_1d, &backend).expect("run sliced ReLU");
    assert_eq!(relu_1d.as_slice(), &[5.0, 6.0, 7.0, 8.0]);

    let relu_2d = coeus_ops::relu(&slice_a, &backend).expect("run strided ReLU");
    assert_eq!(relu_2d.as_slice(), &[6.0, 7.0, 10.0, 11.0]);
}

#[test]
fn test_integer_abs_and_sqrt() {
    let backend = MoiraiBackend::new();
    let data = vec![-5i32, 4, -16, 9];
    let a = Tensor::<i32, MoiraiBackend>::from_slice(vec![4], &data).expect("construct tensor");

    let b = coeus_ops::abs(&a, &backend).expect("run integer absolute value");
    assert_eq!(b.as_slice(), &[5, 4, 16, 9]);

    let c = coeus_ops::sqrt(&b, &backend).expect("run integer square root");
    // sqrt of [5, 4, 16, 9] as integer types:
    // sqrt(5) -> 2.236... cast to i32 -> 2
    // sqrt(4) -> 2
    // sqrt(16) -> 4
    // sqrt(9) -> 3
    assert_eq!(c.as_slice(), &[2, 2, 4, 3]);

    let mut d = a.clone();
    coeus_ops::abs_assign(&mut d, &backend).expect("integer absolute value");
    assert_eq!(d.as_slice(), &[5, 4, 16, 9]);

    coeus_ops::sqrt_assign(&mut d, &backend).expect("integer square root");
    assert_eq!(d.as_slice(), &[2, 2, 4, 3]);
}

#[test]
fn test_sliced_tensor_reshape_and_reduction() {
    let backend = MoiraiBackend::new();

    // Reshape offset contiguous view
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = Tensor::<f32, MoiraiBackend>::from_slice(vec![8], &data).expect("construct tensor");
    let slice_a = a.slice(&[(2, 6)]); // shape [4], offset 2, elements [3, 4, 5, 6]
    assert_eq!(slice_a.layout().offset(), 2);
    assert!(slice_a.is_contiguous());

    let reshaped_a = slice_a.reshape([2, 2]);
    assert_eq!(reshaped_a.shape(), &[2, 2]);
    assert_eq!(reshaped_a.layout().offset(), 2);
    assert!(reshaped_a.is_contiguous());
    assert_eq!(reshaped_a.get(&[0, 0]), 3.0);
    assert_eq!(reshaped_a.get(&[0, 1]), 4.0);
    assert_eq!(reshaped_a.get(&[1, 0]), 5.0);
    assert_eq!(reshaped_a.get(&[1, 1]), 6.0);
    assert_eq!(reshaped_a.as_slice(), &[3.0, 4.0, 5.0, 6.0]);

    // Sum reduction on offset contiguous view
    let sum_val = coeus_ops::sum(&slice_a, &backend).expect("valid sum");
    assert_eq!(sum_val, 18.0); // 3 + 4 + 5 + 6 = 18
}

#[test]
fn host_materialization_respects_view_layout() {
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let tensor = Tensor::<f32, SequentialBackend>::from_slice([3, 4], &values).expect("construct tensor");
    let view = tensor.slice(&[(0, 3), (1, 4)]).transpose();

    assert_eq!(
        view.to_vec().expect("materialize tensor values"),
        vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );
}

#[test]
fn host_cow_borrows_contiguous_and_materializes_strided_storage() {
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let tensor = Tensor::<f32, SequentialBackend>::from_slice([3, 4], &values).expect("construct tensor");
    let contiguous = tensor.slice(&[(1, 3), (0, 4)]);
    let strided = tensor.slice(&[(0, 3), (1, 4)]).transpose();

    assert!(matches!(
        contiguous.host_cow().expect("materialize host view"),
        std::borrow::Cow::Borrowed(_)
    ));
    assert_eq!(
        contiguous.host_cow().expect("materialize host view").as_ref(),
        &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
    assert!(matches!(
        strided.host_cow().expect("materialize host view"),
        std::borrow::Cow::Owned(_)
    ));
    assert_eq!(
        strided.host_cow().expect("materialize host view").as_ref(),
        &[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );
}
