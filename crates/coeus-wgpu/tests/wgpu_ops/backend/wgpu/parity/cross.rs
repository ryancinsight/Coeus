use coeus_tensor::Tensor;

use super::{assert_parity, to_cpu, to_gpu};

#[test]
fn test_wgpu_parity_cross_last_axis() {
    let a = Tensor::from_slice(vec![2, 3], &[1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0])
        .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(vec![2, 3], &[0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0])
        .expect("invariant: test backend operation succeeds");
    // dim=1 is the last axis and both operands are contiguous: exercises the
    // on-device `CrossProductOps` bridge (ADR 0077).
    let cpu = coeus_ops::cross(&a, &b, 1).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::cross(&to_gpu(&a), &to_gpu(&b), 1).expect("invariant: test operation succeeds"),
    );
    assert_parity("cross_last_axis", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_cross_non_last_axis_fallback() {
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
    // cannot express this layout, so `WgpuBackend` falls back to the same
    // shared host-fold every seam-less backend uses (ADR 0077).
    let cpu = coeus_ops::cross(&a, &b, 0).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::cross(&to_gpu(&a), &to_gpu(&b), 0).expect("invariant: test operation succeeds"),
    );
    assert_parity(
        "cross_non_last_axis_fallback",
        cpu.as_slice(),
        gpu.as_slice(),
    );
}
