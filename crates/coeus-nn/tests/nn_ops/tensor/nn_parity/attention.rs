use super::assert_tensor_eq_data;
#[path = "attention/expected.rs"]
mod expected;
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
    let expected_mha_out = expected::mha_out();
    assert_tensor_eq_data(&out_coeus.tensor, &expected_mha_out, 1e-4);

    // Backward
    let loss_coeus = coeus_autograd::sum(&out_coeus);
    loss_coeus
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // Verify input gradients
    let dq_coeus = q_coeus.grad().unwrap();
    let dk_coeus = k_coeus.grad().unwrap();
    let dv_coeus = v_coeus.grad().unwrap();

    let expected_mha_dq = expected::mha_dq();
    let expected_mha_dk = expected::mha_dk();
    let expected_mha_dv = expected::mha_dv();

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

    let expected_mha_dwq = expected::mha_dwq();
    let expected_mha_dbq = expected::mha_dbq();
    let expected_mha_dwk = expected::mha_dwk();
    let expected_mha_dbk = expected::mha_dbk();
    let expected_mha_dwv = expected::mha_dwv();
    let expected_mha_dbv = expected::mha_dbv();
    let expected_mha_dwo = expected::mha_dwo();
    let expected_mha_dbo = expected::mha_dbo();

    let expected_mha_dwq_transposed = expected::transpose_8x8(&expected_mha_dwq);
    let expected_mha_dwk_transposed = expected::transpose_8x8(&expected_mha_dwk);
    let expected_mha_dwv_transposed = expected::transpose_8x8(&expected_mha_dwv);
    let expected_mha_dwo_transposed = expected::transpose_8x8(&expected_mha_dwo);

    assert_tensor_eq_data(&dwq_coeus, &expected_mha_dwq_transposed, 1e-4);
    assert_tensor_eq_data(&dbq_coeus, &expected_mha_dbq, 1e-4);
    assert_tensor_eq_data(&dwk_coeus, &expected_mha_dwk_transposed, 1e-4);
    assert_tensor_eq_data(&dbk_coeus, &expected_mha_dbk, 1e-4);
    assert_tensor_eq_data(&dwv_coeus, &expected_mha_dwv_transposed, 1e-4);
    assert_tensor_eq_data(&dbv_coeus, &expected_mha_dbv, 1e-4);
    assert_tensor_eq_data(&dwo_coeus, &expected_mha_dwo_transposed, 1e-4);
    assert_tensor_eq_data(&dbo_coeus, &expected_mha_dbo, 1e-4);
}
