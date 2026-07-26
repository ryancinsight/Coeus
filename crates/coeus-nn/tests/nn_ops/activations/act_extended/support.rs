pub(crate) fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b} got {a} (diff {:.3e})",
        (a - b).abs()
    );
}

pub(crate) fn assert_close_slice(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (&g, &e) in got.iter().zip(expected.iter()) {
        close(g, e, tol);
    }
}
