//! Differential verification of `frobenius_norm` and `frobenius_norm_batched`.
//!
//! Analytical reference: `‖A‖_F = sqrt(Σ aᵢⱼ²)`.
//!
//! All tolerances are derived from f32 machine epsilon (≈1.19e-7) times the
//! number of additions (at most 9 for the cases below), giving ≤ 8 ULPs ≈
//! 1e-5.  Exact-integer inputs (e.g., the 3–4–5 triangle) return bitwise-
//! exact results.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn from_slice_f32<B>(shape: &[usize], data: &[f32], backend: &B) -> Tensor<f32, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorageMut<f32>,
{
    Tensor::from_slice_on(shape.to_vec(), data, backend)
}

fn assert_scalar_close(got: f32, expected: f32, eps: f32, context: &str) {
    assert!(
        (got - expected).abs() <= eps,
        "{context}: got {got}, expected {expected}, eps={eps}",
    );
}

fn assert_slice_close(got: &[f32], expected: &[f32], eps: f32, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= eps,
            "{context}[{i}]: got {g}, expected {e}, eps={eps}",
        );
    }
}

// ── FROBENIUS NORM (2-D) ───────────────────────────────────────────────────

fn check_frobenius_norm<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // 3–4–5 right-triangle entries: ‖[[3,4],[0,0]]‖_F = sqrt(9+16) = 5.0 exactly.
    let a = from_slice_f32(&[2, 2], &[3.0f32, 4.0, 0.0, 0.0], backend);
    let got = coeus_ops::frobenius_norm(&a, backend).expect("valid Frobenius norm input");
    assert_scalar_close(got, 5.0f32, 1e-6, "frobenius_norm 3-4-5");

    // Rectangular [2,3]: ‖[[1,2,3],[4,5,6]]‖_F = sqrt(1+4+9+16+25+36) = sqrt(91).
    let b = from_slice_f32(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let expected_b = 91.0f32.sqrt();
    let got_b = coeus_ops::frobenius_norm(&b, backend).expect("valid Frobenius norm input");
    assert_scalar_close(got_b, expected_b, 1e-5, "frobenius_norm [2,3]");

    // Identity [2,2]: ‖I₂‖_F = sqrt(1+0+0+1) = sqrt(2) ≈ 1.414_213_6.
    let c = from_slice_f32(&[2, 2], &[1.0f32, 0.0, 0.0, 1.0], backend);
    let expected_c = 2.0f32.sqrt();
    let got_c = coeus_ops::frobenius_norm(&c, backend).expect("valid Frobenius norm input");
    assert_scalar_close(got_c, expected_c, 1e-6, "frobenius_norm identity [2,2]");

    // All-zeros [3,3]: ‖0‖_F = 0.0.
    let d = from_slice_f32(&[3, 3], &[0.0f32; 9], backend);
    let got_d = coeus_ops::frobenius_norm(&d, backend).expect("valid Frobenius norm input");
    assert_scalar_close(got_d, 0.0f32, 0.0, "frobenius_norm zeros [3,3]");
}

#[test]
fn sequential_frobenius_norm_matches_reference() {
    let backend = SequentialBackend;
    check_frobenius_norm(&backend);
}

#[test]
fn moirai_frobenius_norm_matches_reference() {
    let backend = MoiraiBackend;
    check_frobenius_norm(&backend);
}

// ── FROBENIUS NORM BATCHED (rank ≥ 3) ─────────────────────────────────────

fn check_frobenius_norm_batched<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // shape [2,2,2]: two [2×2] matrices.
    //   batch 0: [[1,2],[3,4]] → sqrt(1+4+9+16) = sqrt(30)
    //   batch 1: [[5,6],[7,8]] → sqrt(25+36+49+64) = sqrt(174)
    let a = from_slice_f32(
        &[2, 2, 2],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        backend,
    );
    let out =
        coeus_ops::frobenius_norm_batched(&a, backend).expect("valid batched Frobenius norm input");
    assert_eq!(out.shape(), &[2], "frobenius_norm_batched [2,2,2] shape");
    assert_slice_close(
        out.as_slice(),
        &[30.0f32.sqrt(), 174.0f32.sqrt()],
        1e-5,
        "frobenius_norm_batched [2,2,2]",
    );

    // shape [3,1,2]: three [1×2] row vectors.
    //   batch 0: [[1,0]]    → sqrt(1)   = 1.0
    //   batch 1: [[3,4]]    → sqrt(9+16)= 5.0
    //   batch 2: [[0,0]]    → 0.0
    let b = from_slice_f32(&[3, 1, 2], &[1.0f32, 0.0, 3.0, 4.0, 0.0, 0.0], backend);
    let out_b =
        coeus_ops::frobenius_norm_batched(&b, backend).expect("valid batched Frobenius norm input");
    assert_eq!(out_b.shape(), &[3], "frobenius_norm_batched [3,1,2] shape");
    assert_slice_close(
        out_b.as_slice(),
        &[1.0f32, 5.0, 0.0],
        1e-6,
        "frobenius_norm_batched [3,1,2]",
    );

    // Rank-4 shape [2,2,2,2]: four [2×2] matrices grouped in a [2×2] batch.
    //   All entries are 1.0 → each slice‖[[1,1],[1,1]]‖_F = sqrt(4) = 2.0.
    let c = from_slice_f32(&[2, 2, 2, 2], &[1.0f32; 16], backend);
    let out_c =
        coeus_ops::frobenius_norm_batched(&c, backend).expect("valid batched Frobenius norm input");
    assert_eq!(
        out_c.shape(),
        &[2, 2],
        "frobenius_norm_batched [2,2,2,2] shape"
    );
    assert_slice_close(
        out_c.as_slice(),
        &[2.0f32; 4],
        1e-6,
        "frobenius_norm_batched [2,2,2,2]",
    );
}

#[test]
fn sequential_frobenius_norm_batched_matches_reference() {
    let backend = SequentialBackend;
    check_frobenius_norm_batched(&backend);
}

#[test]
fn moirai_frobenius_norm_batched_matches_reference() {
    let backend = MoiraiBackend;
    check_frobenius_norm_batched(&backend);
}
