use super::{seq, to_cpu, to_gpu};
use coeus_autograd::{rotate_half, sum, Var};
use coeus_tensor::Tensor;

#[test]
fn rotate_half_dispatches_with_wgpu_parity() {
    #[cfg(target_os = "windows")]
    std::env::set_var("WGPU_BACKEND", "dx12");
    if hephaestus_wgpu::WgpuDevice::try_default_with_limits(
        "coeus-wgpu-rotate-half-test",
        wgpu::Limits::default(),
    )
    .is_err()
    {
        assert_ne!(
            std::env::var("HEPHAESTUS_WGPU_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "WGPU CI requires an acquired device"
        );
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
