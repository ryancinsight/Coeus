use super::*;

#[test]
fn test_cuda_parity_cross_last_axis() {
    let Some((s, c)) = backends() else {
        return;
    };
    let a = Tensor::from_slice(vec![2, 3], &[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(vec![2, 3], &[0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0])
        .expect("invariant: test backend operation succeeds");
    // dim=1 is the last axis and both operands are contiguous: exercises the
    // on-device `CrossProductOps` bridge.
    let cpu = coeus_ops::cross(&a, &b, 1).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::cross(&to_gpu(&a, &s, &c), &to_gpu(&b, &s, &c), 1)
            .expect("invariant: test operation succeeds"),
        &c,
        &s,
    );
    assert_parity_tol("cross_last_axis", cpu.as_slice(), gpu.as_slice(), CUDA_TOL);
}

#[test]
fn test_cuda_parity_cross_non_last_axis_fallback() {
    let Some((s, c)) = backends() else {
        return;
    };
    let a = Tensor::from_slice(
        vec![3, 3],
        &[1.0_f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0],
    )
    .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![3, 3],
        &[0.0_f32, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    )
    .expect("invariant: test backend operation succeeds");
    // dim=0 on a [3, 3] tensor is not the last axis: the on-device seam
    // cannot express this layout, so `CudaBackend` falls back to the same
    // shared host-fold every seam-less backend uses.
    let cpu = coeus_ops::cross(&a, &b, 0).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::cross(&to_gpu(&a, &s, &c), &to_gpu(&b, &s, &c), 0)
            .expect("invariant: test operation succeeds"),
        &c,
        &s,
    );
    assert_parity_tol(
        "cross_non_last_axis_fallback",
        cpu.as_slice(),
        gpu.as_slice(),
        CUDA_TOL,
    );
}
