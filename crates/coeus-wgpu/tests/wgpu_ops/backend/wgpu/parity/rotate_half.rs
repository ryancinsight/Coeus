use super::{seq, to_cpu, to_gpu, wgpu};
use coeus_autograd::{rotate_half, sum, Var};
use coeus_tensor::Tensor;

#[test]
fn rotate_half_dispatches_with_wgpu_parity() {
    if !crate::availability::device_available("coeus-wgpu-rotate-half-test") {
        return;
    }
    let cpu = Tensor::from_slice_on(
        [2, 4],
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &seq(),
    );
    let input = Var::new(to_gpu(&cpu), true);
    let output = rotate_half(&input).expect("WGPU rotate-half dispatch");
    sum(&output).backward().expect("WGPU rotate-half backward");
    assert_eq!(
        to_cpu(&output.tensor).as_slice(),
        &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]
    );
    assert_eq!(
        to_cpu(&input.grad().expect("tracked WGPU input gradient")).as_slice(),
        &[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    );
}

#[test]
fn rotate_half_dispatches_with_wgpu_parity_f64_forward() {
    if !crate::availability::device_supports_f64("coeus-wgpu-rotate-half-test-f64") {
        return;
    }
    // Forward-only: f64 backward on WGPU waits on the six core `BackendOps`
    // impls admitting f64 (all still `WgpuScalar`-gated); the Tensor-level
    // forward needs only `RotateHalfOps<T>`, which is now generic.
    let cpu = Tensor::from_slice_on(
        [2, 4],
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &seq(),
    );
    let gpu = cpu.to_backend_on(&seq(), &wgpu());
    let output = coeus_ops::rotate_half(&gpu, &wgpu()).expect("WGPU rotate-half dispatch (f64)");
    assert_eq!(
        output.to_backend_on(&wgpu(), &seq()).as_slice(),
        &[-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]
    );
}
