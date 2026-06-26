//! Differential verification of glu, causal_softmax, and topk.
//!
//! References derive from closed-form definitions:
//! - `glu`:            first_half * sigmoid(second_half);  sigmoid(x) = 1/(1+exp(-x))
//! - `causal_softmax`: lower-triangular masked softmax; masked positions emit 0.0
//! - `topk`:           k largest (or smallest) values sorted by value, descending (ascending)
//!
//! All tolerances are derived from f32 machine epsilon (≈1.19e-7):
//! transcendental chains (exp, div, mul) accumulate at most ~8 ULPs ≈ 1e-6 for f32.
//! Exact integer values (topk indices, exactly-representable floats) are bit-equal.

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

fn assert_close(got: &[f32], expected: &[f32], eps: f32, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= eps || (g.is_nan() && e.is_nan()),
            "{context}[{i}]: got {g}, expected {e}, eps={eps}",
        );
    }
}

fn assert_eq_i64(got: &[i64], expected: &[i64], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(g, e, "{context}[{i}]: got {g}, expected {e}");
    }
}

// ── GLU ────────────────────────────────────────────────────────────────────

fn check_glu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // Input [[1.0, 2.0, 3.0, 4.0]] shape [1,4]; dim=1.
    // first_half = [[1.0, 2.0]], second_half = [[3.0, 4.0]]
    // sigmoid(3.0) ≈ 0.9525741, sigmoid(4.0) ≈ 0.9820138
    // result = [1.0 * 0.9525741, 2.0 * 0.9820138] ≈ [0.9525741, 1.9640276]
    let x = from_slice_f32(&[1, 4], &[1.0f32, 2.0, 3.0, 4.0], backend);
    let out = coeus_ops::glu(&x, 1, backend);
    assert_eq!(out.shape(), &[1, 2]);
    assert_close(
        out.as_slice(),
        &[0.952_574_1f32, 1.964_027_6],
        1e-5,
        "glu [1,4] dim=1",
    );

    // 2-row: shape [2, 4].
    // Row 0: halves [1.0, 2.0] and [0.0, 0.0]
    //   sigmoid(0.0) = 0.5 → [0.5, 1.0]
    // Row 1: halves [4.0, 4.0] and [10.0, 10.0]
    //   sigmoid(10.0) ≈ 0.9999546 → [3.9998185, 3.9998185]
    let x2 = from_slice_f32(
        &[2, 4],
        &[1.0f32, 2.0, 0.0, 0.0, 4.0, 4.0, 10.0, 10.0],
        backend,
    );
    let out2 = coeus_ops::glu(&x2, 1, backend);
    assert_eq!(out2.shape(), &[2, 2]);
    assert_close(
        out2.as_slice(),
        &[0.5f32, 1.0, 3.999_818_5, 3.999_818_5],
        1e-5,
        "glu [2,4] dim=1",
    );
}

#[test]
fn sequential_glu_matches_reference() {
    let backend = SequentialBackend;
    check_glu(&backend);
}

#[test]
fn moirai_glu_matches_reference() {
    let backend = MoiraiBackend;
    check_glu(&backend);
}

// ── CAUSAL SOFTMAX ─────────────────────────────────────────────────────────

fn check_causal_softmax<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    // Input [[0.0, 1.0], [2.0, 3.0]] shape [2,2]; dim=1 (final axis).
    // Causal mask for shape [2,2]: row i keeps j <= i.
    //   Row 0 (i=0): only j=0 unmasked → softmax([0.0]) = [1.0]; j=1 → 0.0
    //   Row 1 (i=1): j=0,1 unmasked → softmax([2.0, 3.0])
    //     row_max=3.0; e^(2-3)=e^-1≈0.36787944, e^0=1.0; sum≈1.36787944
    //     → [0.26894142, 0.73105858]
    let x = from_slice_f32(&[2, 2], &[0.0f32, 1.0, 2.0, 3.0], backend);
    let out = coeus_ops::causal_softmax(&x, 1, backend);
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(
        out.as_slice(),
        &[1.0f32, 0.0, 0.268_941_4, 0.731_058_6],
        1e-6,
        "causal_softmax [2,2] dim=1",
    );

    // 3-row [3,3]: tokens q=3, k=3.
    // All identical inputs = 1.0 → each row softmax over unmasked prefix gives uniform.
    //   Row 0: [1.0, 0.0, 0.0]
    //   Row 1: [0.5, 0.5, 0.0]
    //   Row 2: [1/3, 1/3, 1/3]
    let x3 = from_slice_f32(&[3, 3], &[1.0f32; 9], backend);
    let out3 = coeus_ops::causal_softmax(&x3, 1, backend);
    assert_eq!(out3.shape(), &[3, 3]);
    let third = 1.0f32 / 3.0;
    assert_close(
        out3.as_slice(),
        &[1.0f32, 0.0, 0.0, 0.5, 0.5, 0.0, third, third, third],
        1e-6,
        "causal_softmax [3,3] uniform dim=1",
    );
}

#[test]
fn sequential_causal_softmax_matches_reference() {
    let backend = SequentialBackend;
    check_causal_softmax(&backend);
}

#[test]
fn moirai_causal_softmax_matches_reference() {
    let backend = MoiraiBackend;
    check_causal_softmax(&backend);
}

// ── TOPK ───────────────────────────────────────────────────────────────────

fn check_topk<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f32> + coeus_ops::BackendOps<i64> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // Input [[3.0, 1.0, 4.0, 5.0, 2.0], [9.0, 2.0, 6.0, 5.0, 3.0]] shape [2,5].
    // topk(k=2, dim=1, largest=true): top-2 values sorted descending.
    //   Row 0: values=[5.0, 4.0], indices=[3, 2]
    //   Row 1: values=[9.0, 6.0], indices=[0, 2]
    let x = from_slice_f32(
        &[2, 5],
        &[3.0f32, 1.0, 4.0, 5.0, 2.0, 9.0, 2.0, 6.0, 5.0, 3.0],
        backend,
    );

    let (val, idx) = coeus_ops::topk(&x, 2, 1, true);
    assert_eq!(val.shape(), &[2, 2]);
    assert_eq!(idx.shape(), &[2, 2]);
    assert_close(
        val.as_slice(),
        &[5.0f32, 4.0, 9.0, 6.0],
        0.0,
        "topk largest values",
    );
    assert_eq_i64(idx.as_slice(), &[3, 2, 0, 2], "topk largest indices");

    // topk(k=2, dim=1, largest=false): bottom-2 values sorted ascending.
    //   Row 0: values=[1.0, 2.0], indices=[1, 4]
    //   Row 1: values=[2.0, 3.0], indices=[1, 4]
    let (val2, idx2) = coeus_ops::topk(&x, 2, 1, false);
    assert_eq!(val2.shape(), &[2, 2]);
    assert_close(
        val2.as_slice(),
        &[1.0f32, 2.0, 2.0, 3.0],
        0.0,
        "topk smallest values",
    );
    assert_eq_i64(idx2.as_slice(), &[1, 4, 1, 4], "topk smallest indices");

    // k=1 edge: top-1 per row.
    let (val1, idx1) = coeus_ops::topk(&x, 1, 1, true);
    assert_eq!(val1.shape(), &[2, 1]);
    assert_close(val1.as_slice(), &[5.0f32, 9.0], 0.0, "topk k=1 values");
    assert_eq_i64(idx1.as_slice(), &[3, 0], "topk k=1 indices");
}

#[test]
fn sequential_topk_matches_reference() {
    let backend = SequentialBackend;
    check_topk(&backend);
}

#[test]
fn moirai_topk_matches_reference() {
    let backend = MoiraiBackend;
    check_topk(&backend);
}
