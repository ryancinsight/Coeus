// Differential correctness for the sparse ops (SpMV / SpMM) against a dense
// matmul oracle.
//
// `coeus_ops::{spmv, spmm}` operate on a CSR matrix; the same product computed
// by densifying the matrix and running `coeus_ops::matmul` is the analytical
// reference. Equality (within f32 roundoff — the sparse path skips structural
// zeros, so accumulation order differs) verifies the CSR traversal and the
// dense<->CSR conversion together.

use coeus_core::SequentialBackend;
use coeus_ops::{dense_to_csr, matmul, spmm, spmv};
use coeus_tensor::Tensor;

type Seq = SequentialBackend;

/// A known 4x5 matrix with a realistic scatter of structural zeros.
fn dense_4x5() -> Tensor<f32, Seq> {
    #[rustfmt::skip]
    let data = vec![
        1.0f32, 0.0, 2.0, 0.0, 0.0,
        0.0,    3.0, 0.0, 0.0, 4.0,
        0.0,    0.0, 0.0, 0.0, 0.0, // fully empty row exercises row_offsets
        5.0,    0.0, 0.0, 6.0, 7.0,
    ];
    Tensor::<f32, Seq>::from_slice([4, 5], &data)
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 * (1.0 + w.abs());
        assert!((g - w).abs() <= tol, "{label}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn spmv_matches_dense_matmul() {
    let s = SequentialBackend::new();
    let dense = dense_4x5();
    let csr = dense_to_csr(&dense, &s);

    // x: [5] vector; reference reshapes to [5,1] for the dense matmul.
    let x_data = [0.5f32, -1.0, 2.0, 3.0, -0.25];
    let x = Tensor::<f32, Seq>::from_slice([5], &x_data);
    let x_col = Tensor::<f32, Seq>::from_slice([5, 1], &x_data);

    let y_sparse = spmv(&csr, &x, &s); // [4]
    let y_dense = matmul(&dense, &x_col, &s); // [4,1]

    assert_eq!(y_sparse.shape(), &[4]);
    assert_close("spmv", y_sparse.as_slice(), y_dense.as_slice());
}

#[test]
fn spmm_matches_dense_matmul() {
    let s = SequentialBackend::new();
    let dense = dense_4x5();
    let csr = dense_to_csr(&dense, &s);

    // B: [5, 3] dense right operand.
    let b_data: Vec<f32> = (0..5 * 3).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let b = Tensor::<f32, Seq>::from_slice([5, 3], &b_data);

    let c_sparse = spmm(&csr, &b, &s); // [4, 3]
    let c_dense = matmul(&dense, &b, &s); // [4, 3]

    assert_eq!(c_sparse.shape(), &[4, 3]);
    assert_close("spmm", c_sparse.as_slice(), c_dense.as_slice());
}

#[test]
fn spmm_identity_roundtrip() {
    // A·I = A: SpMM against the identity recovers the densified CSR matrix,
    // independently checking dense_to_csr fidelity.
    let s = SequentialBackend::new();
    let dense = dense_4x5();
    let csr = dense_to_csr(&dense, &s);

    let mut eye = vec![0.0f32; 5 * 5];
    for i in 0..5 {
        eye[i * 5 + i] = 1.0;
    }
    let identity = Tensor::<f32, Seq>::from_slice([5, 5], &eye);

    let c = spmm(&csr, &identity, &s); // [4, 5] == dense
    assert_close("spmm_identity", c.as_slice(), dense.as_slice());
}

// ── SpMM backward gradients vs dense analytical oracle ──

#[test]
fn spmm_backward_dense_matches_transpose_matmul() {
    // grad_B = Aᵀ · grad_C. Reference: transpose the densified A and matmul.
    use coeus_ops::spmm_backward_dense;
    let s = SequentialBackend::new();
    let dense = dense_4x5(); // A: [M=4, K=5]
    let (m, k, n) = (4usize, 5usize, 3usize);
    let csr = dense_to_csr(&dense, &s);

    let gc: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let grad_c = Tensor::<f32, Seq>::from_slice([m, n], &gc);

    let grad_b = spmm_backward_dense(
        csr.values(),
        csr.col_indices(),
        csr.row_offsets(),
        &[m, k],
        &grad_c,
        &s,
    ); // [K, N]

    // Reference: Aᵀ [K, M] · grad_C [M, N].
    let a = dense.as_slice();
    let mut at = vec![0.0f32; k * m];
    for r in 0..m {
        for c in 0..k {
            at[c * m + r] = a[r * k + c];
        }
    }
    let a_t = Tensor::<f32, Seq>::from_slice([k, m], &at);
    let grad_b_ref = matmul(&a_t, &grad_c, &s); // [K, N]

    assert_eq!(grad_b.shape(), &[k, n]);
    assert_close("spmm_bwd_dense", grad_b.as_slice(), grad_b_ref.as_slice());
}

#[test]
fn spmm_backward_values_matches_masked_outer() {
    // grad_value at stored (r, col) = Σₙ grad_C[r,n] · B[col,n], returned in the
    // CSR value order. Reference computed directly from the CSR structure.
    use coeus_ops::spmm_backward_values;
    let s = SequentialBackend::new();
    let dense = dense_4x5(); // A: [M=4, K=5]
    let (m, k, n) = (4usize, 5usize, 3usize);
    let csr = dense_to_csr(&dense, &s);

    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.15 - 0.3).collect();
    let b = Tensor::<f32, Seq>::from_slice([k, n], &b_data);
    let gc: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let grad_c = Tensor::<f32, Seq>::from_slice([m, n], &gc);

    let grad_vals = spmm_backward_values(
        csr.col_indices(),
        csr.row_offsets(),
        &[m, k],
        &b,
        &grad_c,
        &s,
    ); // [nnz]

    // Reference in CSR value order.
    let row_off: Vec<i64> = csr.row_offsets().as_slice().to_vec();
    let cols: Vec<i64> = csr.col_indices().as_slice().to_vec();
    let nnz = cols.len();
    let mut reference = vec![0.0f32; nnz];
    for r in 0..m {
        for i in row_off[r] as usize..row_off[r + 1] as usize {
            let col = cols[i] as usize;
            let mut g = 0.0f32;
            for j in 0..n {
                g += gc[r * n + j] * b_data[col * n + j];
            }
            reference[i] = g;
        }
    }

    assert_eq!(grad_vals.as_slice().len(), nnz);
    assert_close("spmm_bwd_values", grad_vals.as_slice(), &reference);
}
