use super::assert_tensor_eq_data;
use coeus_autograd::Var as CoeusVar;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor as CoeusTensor;

#[test]
fn test_mha_parity() {
    let backend = SequentialBackend::new();

    // Query, key, value shape: [2, 3, 8] (batch=2, seq=3, d_model=8)
    // Heads = 2, so H = 2.

    let q_data = vec![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7,
    ];
    let k_data = q_data.clone();
    let v_data = q_data.clone();

    let wq_data = vec![0.1f32; 64];
    let bq_data = vec![0.05f32; 8];
    let wk_data = vec![0.2f32; 64];
    let bk_data = vec![0.1f32; 8];
    let wv_data = vec![0.3f32; 64];
    let bv_data = vec![0.15f32; 8];
    let wo_data = vec![0.4f32; 64];
    let bo_data = vec![0.2f32; 8];

    // Coeus setup
    let q_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &q_data),
        true,
    );
    let k_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &k_data),
        true,
    );
    let v_coeus = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3, 8], &v_data),
        true,
    );

    let mut mha_coeus = coeus_nn::MultiHeadAttention::<f32, SequentialBackend, 2>::new(8, true);
    mha_coeus.w_q = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wq_data),
        true,
    );
    mha_coeus.b_q = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bq_data),
        true,
    ));
    mha_coeus.w_k = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wk_data),
        true,
    );
    mha_coeus.b_k = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bk_data),
        true,
    ));
    mha_coeus.w_v = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wv_data),
        true,
    );
    mha_coeus.b_v = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bv_data),
        true,
    ));
    mha_coeus.w_o = CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8, 8], &wo_data),
        true,
    );
    mha_coeus.b_o = Some(CoeusVar::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![8], &bo_data),
        true,
    ));

    let out_coeus = mha_coeus.forward_cross(&q_coeus, &k_coeus, &v_coeus, None);

    // Verify Q, K, V projections match
    let q_proj_coeus = {
        let flat = coeus_autograd::reshape(&q_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_q);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_q.as_ref().unwrap())
    };
    let k_proj_coeus = {
        let flat = coeus_autograd::reshape(&k_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_k);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_k.as_ref().unwrap())
    };
    let v_proj_coeus = {
        let flat = coeus_autograd::reshape(&v_coeus, [6, 8]);
        let w_t = coeus_autograd::transpose_2d(&mha_coeus.w_v);
        let out_flat = coeus_autograd::matmul(&flat, &w_t);
        coeus_autograd::add(&out_flat, mha_coeus.b_v.as_ref().unwrap())
    };

    let q_split = coeus_autograd::reshape(&q_proj_coeus, [2, 3, 2, 4]);
    let q_perm = coeus_autograd::permute(&q_split, &[0, 2, 1, 3]);
    let q_heads = coeus_autograd::reshape(&q_perm, [4, 3, 4]);

    let k_split = coeus_autograd::reshape(&k_proj_coeus, [2, 3, 2, 4]);
    let k_perm = coeus_autograd::permute(&k_split, &[0, 2, 1, 3]);
    let k_heads = coeus_autograd::reshape(&k_perm, [4, 3, 4]);

    let v_split = coeus_autograd::reshape(&v_proj_coeus, [2, 3, 2, 4]);
    let v_perm = coeus_autograd::permute(&v_split, &[0, 2, 1, 3]);
    let v_heads = coeus_autograd::reshape(&v_perm, [4, 3, 4]);

    let (out_tensor, _attn_weights_coeus) = coeus_ops::scaled_dot_product_attention(
        &q_heads.tensor,
        &k_heads.tensor,
        &v_heads.tensor,
        None,
        false,
        0.5f32,
        &backend,
    );

    let out_var = CoeusVar::new(out_tensor, false);
    let merged_split = coeus_autograd::reshape(&out_var, [2, 2, 3, 4]);
    let merged_perm = coeus_autograd::permute(&merged_split, &[0, 2, 1, 3]);
    let _merged = coeus_autograd::reshape(&merged_perm, [2, 3, 8]);

    // Verify forward output
    let expected_mha_out = vec![
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        5.160251f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        0.535744f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        6.037920f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        -8.599031f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        11.040882f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
        -9.066002f32,
    ];
    assert_tensor_eq_data(&out_coeus.tensor, &expected_mha_out, 1e-4);

    // Backward
    let loss_coeus = coeus_autograd::sum(&out_coeus);
    loss_coeus.backward();

    // Verify input gradients
    let dq_coeus = q_coeus.grad().unwrap();
    let dk_coeus = k_coeus.grad().unwrap();
    let dv_coeus = v_coeus.grad().unwrap();

    let expected_mha_dq = vec![
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        3.055613f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        5.645340f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.507951f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        1.367525f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.041598f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
        0.680567f32,
    ];
    let expected_mha_dk = vec![
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        -2.527703f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        0.197418f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        2.330285f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -3.211075f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        -0.106984f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
        3.318061f32,
    ];
    let expected_mha_dv = vec![
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        6.522905f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        5.600709f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        10.916387f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        3.288871f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        7.702006f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
        12.049126f32,
    ];

    assert_tensor_eq_data(&dq_coeus, &expected_mha_dq, 1e-4);
    assert_tensor_eq_data(&dk_coeus, &expected_mha_dk, 1e-4);
    assert_tensor_eq_data(&dv_coeus, &expected_mha_dv, 1e-4);

    // Verify parameter gradients (note: PyTorch weight matrices are transposed compared to Coeus)
    let dwq_coeus = mha_coeus.w_q.grad().unwrap();
    let dbq_coeus = mha_coeus.b_q.as_ref().unwrap().grad().unwrap();
    let dwk_coeus = mha_coeus.w_k.grad().unwrap();
    let dbk_coeus = mha_coeus.b_k.as_ref().unwrap().grad().unwrap();
    let dwv_coeus = mha_coeus.w_v.grad().unwrap();
    let dbv_coeus = mha_coeus.b_v.as_ref().unwrap().grad().unwrap();
    let dwo_coeus = mha_coeus.w_o.grad().unwrap();
    let dbo_coeus = mha_coeus.b_o.as_ref().unwrap().grad().unwrap();

    let expected_mha_dwq = vec![
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.034660f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.420694f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -1.806727f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.192761f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.578795f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -2.964828f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350863f32,
        -3.350862f32,
        -3.350862f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
        -3.736896f32,
    ];
    let expected_mha_dbq = vec![
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
        15.373242f32,
    ];
    let expected_mha_dwk = vec![
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.579298f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.617349f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.655399f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.693449f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.731500f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.769550f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807601f32,
        -0.807602f32,
        -0.807602f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
        -0.845651f32,
    ];
    let expected_mha_dbk = vec![
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
        0.000001f32,
    ];
    let expected_mha_dwv = vec![
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.183809f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        -0.008701f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.166407f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341516f32,
        0.341515f32,
        0.341515f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.516623f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691731f32,
        0.691730f32,
        0.691730f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        0.866839f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041946f32,
        1.041947f32,
        1.041947f32,
    ];
    let expected_mha_dbv = vec![
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
        19.200001f32,
    ];
    let expected_mha_dwo = vec![
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
        1.221802f32,
    ];
    let expected_mha_dbo = vec![6.0f32, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0];

    // Note: expected_mha_dwq, dwk, dwv, dwo in expected_values.txt are pre-transposed from PyTorch.
    // Let's transpose PyTorch expected to match Coeus's weight layout.
    // For MHA, the weights are stored as [8, 8] in both, but Coeus does flat @ w_q.T or w_q.
    // In our manual projection: w_t = transpose_2d(&mha_coeus.w_q) -> flat @ w_t
    // So the weight matrix in Coeus has shape [8, 8], where row-major is [in_features, out_features] or [out_features, in_features]
    // Let's verify by transposing the expected 8x8 matrix if needed.
    // The expected_mha_dwq from PyTorch was 8x8, where output was printed as row-major.
    // Let's check: PyTorch weight grad is shape [8, 8], and we printed it.
    // Coeus weight grad is compared to `dwq_burn.transpose()`, which means Coeus weight layout is the transpose of PyTorch's.
    // Therefore, in our self-contained test, we should transpose the expected PyTorch array to match Coeus weight grad!
    // PyTorch weight grad shape is [8, 8]. If we transpose it, the element at [row, col] goes to [col, row].
    // Let's write a simple helper to assert with transposition, or pre-transpose the vectors!
    // Let's write a transpose helper:
    let expected_mha_dwq_transposed = transpose_8x8(&expected_mha_dwq);
    let expected_mha_dwk_transposed = transpose_8x8(&expected_mha_dwk);
    let expected_mha_dwv_transposed = transpose_8x8(&expected_mha_dwv);
    let expected_mha_dwo_transposed = transpose_8x8(&expected_mha_dwo);

    assert_tensor_eq_data(&dwq_coeus, &expected_mha_dwq_transposed, 1e-4);
    assert_tensor_eq_data(&dbq_coeus, &expected_mha_dbq, 1e-4);
    assert_tensor_eq_data(&dwk_coeus, &expected_mha_dwk_transposed, 1e-4);
    assert_tensor_eq_data(&dbk_coeus, &expected_mha_dbk, 1e-4);
    assert_tensor_eq_data(&dwv_coeus, &expected_mha_dwv_transposed, 1e-4);
    assert_tensor_eq_data(&dbv_coeus, &expected_mha_dbv, 1e-4);
    assert_tensor_eq_data(&dwo_coeus, &expected_mha_dwo_transposed, 1e-4);
    assert_tensor_eq_data(&dbo_coeus, &expected_mha_dbo, 1e-4);
}

fn transpose_8x8(src: &[f32]) -> Vec<f32> {
    assert_eq!(src.len(), 64);
    let mut dst = vec![0.0f32; 64];
    for r in 0..8 {
        for c in 0..8 {
            dst[c * 8 + r] = src[r * 8 + c];
        }
    }
    dst
}
