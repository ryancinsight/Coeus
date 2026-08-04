use super::{assert_parity, seq, to_cpu, to_gpu};
use coeus_autograd::Var;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackendError;

#[test]
fn cross_entropy_dispatches_with_wgpu_value_and_gradient_parity() {
    #[cfg(target_os = "windows")]
    std::env::set_var("WGPU_BACKEND", "dx12");
    if hephaestus_wgpu::WgpuDevice::try_default_with_limits(
        "coeus-wgpu-cross-entropy-test",
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

    let logits = Tensor::from_slice_on(
        [3, 4],
        &[
            1.5_f32, 0.5, -0.5, 0.25, -1.0, 2.0, 0.0, 0.75, 3.0, -2.0, 1.0, 0.0,
        ],
        &seq(),
    );
    let cpu_logits = Var::new(logits.clone(), true);
    let wgpu_logits = Var::new(to_gpu(&logits), true);
    let targets = [0_usize, 1, 2];

    let invalid_target = coeus_nn::cross_entropy_loss(&wgpu_logits, &[0, 1, 4]);
    assert!(matches!(
        invalid_target,
        Err(WgpuBackendError::Validation(
            coeus_core::BackendError::IndexOutOfRange {
                position: 2,
                index: 4,
                bound: 4,
                ..
            }
        ))
    ));

    let cpu_loss = coeus_nn::cross_entropy_loss(&cpu_logits, &targets)
        .expect("CPU cross-entropy dispatch must succeed");
    let wgpu_loss = coeus_nn::cross_entropy_loss(&wgpu_logits, &targets)
        .expect("WGPU cross-entropy dispatch must succeed");
    cpu_loss
        .backward()
        .expect("CPU cross-entropy backward must succeed");
    wgpu_loss
        .backward()
        .expect("WGPU cross-entropy backward must succeed");

    let wgpu_loss = to_cpu(&wgpu_loss.tensor);
    let cpu_gradient = cpu_logits.grad().expect("tracked CPU logits gradient");
    let wgpu_gradient = to_cpu(&wgpu_logits.grad().expect("tracked WGPU logits gradient"));
    assert_parity(
        "cross-entropy loss",
        cpu_loss.tensor.as_slice(),
        wgpu_loss.as_slice(),
    );
    assert_parity(
        "cross-entropy logits gradient",
        cpu_gradient.as_slice(),
        wgpu_gradient.as_slice(),
    );
}
