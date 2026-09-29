//! Tests for the norm reductions in [`super`].

use super::*;
use coeus_core::SequentialBackend;

fn v3() -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice(vec![5], &[1.0f64, -2.0, 3.0, -4.0, 5.0])
}

fn ref_p(x: &[f64], p: f64) -> f64 {
    let s: f64 = x.iter().map(|&v| v.abs().powf(p)).sum();
    s.powf(1.0 / p)
}

#[test]
fn norm_p_p2_matches_classical_l2() {
    let b = SequentialBackend::new();
    let x = v3();
    let got = norm_p(&x, 2.0_f64, &b);
    let want = (55.0_f64).sqrt();
    assert!(
        (got - want).abs() < 1e-12,
        "norm_p(p=2) = {got}, want {want}"
    );
}

#[test]
fn norm_p_p1_matches_manhattan_distance() {
    let b = SequentialBackend::new();
    let x = v3();
    let got = norm_p(&x, 1.0_f64, &b);
    let want = 15.0_f64;
    assert!(
        (got - want).abs() < 1e-12,
        "norm_p(p=1) = {got}, want {want}"
    );
}

#[test]
fn norm_p_p3_matches_cubic_reference() {
    let b = SequentialBackend::new();
    let x = v3();
    let got = norm_p(&x, 3.0_f64, &b);
    let want = ref_p(&[1.0, -2.0, 3.0, -4.0, 5.0], 3.0);
    assert!(
        (got - want).abs() < 1e-10,
        "norm_p(p=3) = {got}, want {want}"
    );
}

#[test]
fn norm_p_is_identical_to_norm_at_p2() {
    let b = SequentialBackend::new();
    let x = v3();
    let n = norm(&x, &b).expect("valid norm test input");
    let n_p = norm_p(&x, 2.0_f64, &b);
    assert_eq!(n.to_bits(), n_p.to_bits());
}

#[test]
#[should_panic(expected = "empty tensor has no norm")]
fn norm_p_empty_panics() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(vec![0], &[0.0f64; 0]);
    let _ = norm_p(&x, 2.0_f64, &b);
}

#[test]
#[should_panic(expected = "ord must be a finite positive number")]
fn norm_p_negative_ord_panics() {
    let b = SequentialBackend::new();
    let x = v3();
    let _ = norm_p(&x, -1.0_f64, &b);
}

#[test]
#[should_panic(expected = "ord must be a finite positive number")]
fn norm_p_zero_ord_panics() {
    let b = SequentialBackend::new();
    let x = v3();
    let _ = norm_p(&x, 0.0_f64, &b);
}

#[test]
#[should_panic(expected = "ord must be a finite positive number")]
fn norm_p_infinite_ord_panics() {
    let b = SequentialBackend::new();
    let x = v3();
    let _ = norm_p(&x, f64::INFINITY, &b);
}

#[test]
fn norm_p_axis_axis1_matches_row_references() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 3],
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
    );
    let got = norm_p_axis(&x, 2.0, 1, &b);
    let want = [
        (1.0_f64 + 4.0 + 9.0).sqrt(),
        (16.0_f64 + 25.0 + 36.0).sqrt(),
    ];
    assert_eq!(got.shape(), &[2, 1]);
    assert!(
        got.as_slice()
            .iter()
            .zip(want)
            .all(|(&g, w)| (g - w).abs() < 1e-12),
        "norm_p_axis(axis=1) = {:?}, want {:?}",
        got.as_slice(),
        want
    );
}

#[test]
fn norm_p_axis_axis0_matches_column_references() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 3],
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
    );
    let got = norm_p_axis(&x, 1.0, 0, &b);
    let want = [5.0, 7.0, 9.0];
    assert_eq!(got.shape(), &[1, 3]);
    assert_eq!(got.as_slice(), &want);
}

#[test]
fn norm_p_axis_rank1_reduces_to_scalar_tensor() {
    let b = SequentialBackend::new();
    let x = v3();
    let got = norm_p_axis(&x, 2.0, 0, &b);
    let n_global = norm_p(&x, 2.0, &b);
    assert_eq!(got.shape(), &[1]);
    assert_eq!(got.as_slice()[0].to_bits(), n_global.to_bits());
}

#[test]
fn norm_p_axis_3d_axis1_matches_manual_per_slice() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 3, 2],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0,
        ],
    );
    let got = norm_p_axis(&x, 3.0, 1, &b);
    assert_eq!(got.shape(), &[2, 1, 2]);
    let want = [
        (1.0_f64 + 27.0 + 125.0).cbrt(),
        (8.0_f64 + 64.0 + 216.0).cbrt(),
        (1.0_f64 + 27.0 + 125.0).cbrt(),
        (8.0_f64 + 64.0 + 216.0).cbrt(),
    ];
    for (g, w) in got.as_slice().iter().zip(want) {
        assert!(
            (*g - w).abs() < 1e-9,
            "norm_p_axis(3D, axis=1) = {g}, want {w}"
        );
    }
}

#[test]
fn norm_p_accepts_zero_copy_strided_views() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 3],
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
    );
    let shared = x.clone();
    let transposed = x.permute(&[1, 0]);
    let got = norm_p(&transposed, 2.0, &b);
    let want = (91.0_f64).sqrt();
    assert!((got - want).abs() < 1e-12);

    let rows = norm_p_axis(&transposed, 2.0, 1, &b);
    let expected = [17.0_f64.sqrt(), 29.0_f64.sqrt(), 45.0_f64.sqrt()];
    assert_eq!(rows.shape(), &[3, 1]);
    for (&actual, expected) in rows.as_slice().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert_eq!(shared.as_slice(), &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
}

#[test]
#[should_panic(expected = "axis 2 out of bounds")]
fn norm_p_axis_out_of_range_axis_panics() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 3], &[1.0; 6]);
    let _ = norm_p_axis(&x, 2.0, 2, &b);
}

#[test]
#[should_panic(expected = "axis 1 has zero elements")]
fn norm_p_axis_zero_size_axis_panics() {
    let b = SequentialBackend::new();
    let x = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 0, 3], &[]);
    let _ = norm_p_axis(&x, 2.0, 1, &b);
}

#[test]
#[should_panic(expected = "ord must be a finite positive number")]
fn norm_p_axis_non_positive_ord_panics() {
    let b = SequentialBackend::new();
    let x = v3();
    let _ = norm_p_axis(&x, 0.0, 0, &b);
}

// ── frobenius_norm / frobenius_norm_batched ─────────────────────────────

fn mat3x3(data: &[f64; 9]) -> Tensor<f64, SequentialBackend> {
    Tensor::<f64, SequentialBackend>::from_slice(vec![3, 3], data)
}

#[test]
fn frobenius_norm_2d_matches_torch_oracle() {
    let b = SequentialBackend::new();
    let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let got = frobenius_norm(&a, &b).expect("valid Frobenius norm test input");
    let want = (204.0_f64).sqrt();
    assert!(
        (got - want).abs() < 1e-12,
        "frobenius_norm(3x3) = {got}, want {want}"
    );
}

#[test]
fn frobenius_norm_2d_identity_matrix_is_sqrt_3() {
    let b = SequentialBackend::new();
    let id = mat3x3(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let got = frobenius_norm(&id, &b).expect("valid Frobenius norm test input");
    let want = 3.0_f64.sqrt();
    assert!(
        (got - want).abs() < 1e-12,
        "frobenius_norm(I_3) = {got}, want {want}"
    );
}

#[test]
fn frobenius_norm_batched_3d_returns_per_batch_scalars() {
    let b = SequentialBackend::new();
    let a = mat3x3(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let stacked = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 3, 3],
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0,
        ],
    );
    let got =
        frobenius_norm_batched(&stacked, &b).expect("valid batched Frobenius norm test input");
    assert_eq!(got.shape(), &[2]);
    let want = (204.0_f64).sqrt();
    for g in got.as_slice() {
        assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
    }
    let ref_scalar = frobenius_norm(&a, &b).expect("valid Frobenius norm test input");
    assert!(
        (ref_scalar - want).abs() < 1e-12,
        "2-D refr scalar = {ref_scalar}, want {want}"
    );
}

#[test]
fn frobenius_norm_batched_4d_collapses_last_two_dims() {
    let b = SequentialBackend::new();
    let batch = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 2, 3, 3],
        &[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    );
    let got =
        frobenius_norm_batched(&batch, &b).expect("valid batched Frobenius norm test input");
    assert_eq!(got.shape(), &[2, 2]);
    let want = 3.0_f64.sqrt();
    for g in got.as_slice() {
        assert!((*g - want).abs() < 1e-12, "got {g}, want {want}");
    }
}

#[test]
fn frobenius_norm_batched_strided_matches_analytical_reference() {
    let b = SequentialBackend::new();
    let source = Tensor::<f64, SequentialBackend>::from_slice(
        vec![2, 2, 3],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
        ],
    );
    let strided = source.permute(&[0, 2, 1]);
    let got = frobenius_norm_batched(&strided, &b).expect("valid strided norm input");

    assert_eq!(got.shape(), &[2]);
    let want = 91.0_f64.sqrt();
    assert!(
        got.as_slice()
            .iter()
            .all(|value| (*value - want).abs() < 1e-12),
        "strided Frobenius norms = {:?}, want [{want}, {want}]",
        got.as_slice()
    );
    assert_eq!(
        source.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    );
}

#[test]
fn frobenius_norm_batched_2d_returns_zero_dim_scalar_tensor() {
    let b = SequentialBackend::new();
    let a = mat3x3(&[3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0]);
    let got = frobenius_norm_batched(&a, &b).expect("valid batched Frobenius norm test input");
    assert_eq!(got.shape(), &[]);
    let want = 50.0_f64.sqrt();
    let v = got.as_slice()[0];
    assert!((v - want).abs() < 1e-12, "got {v}, want {want}");
}

#[test]
#[should_panic(expected = "rank >= 2")]
fn frobenius_norm_batched_1d_panics() {
    let b = SequentialBackend::new();
    let x = v3();
    let _ = frobenius_norm_batched(&x, &b);
}
