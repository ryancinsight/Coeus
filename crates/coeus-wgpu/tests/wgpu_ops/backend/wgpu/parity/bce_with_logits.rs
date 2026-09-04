use super::{assert_parity, seq, to_cpu, to_gpu};
use coeus_autograd::Var;
use coeus_tensor::Tensor;

#[test]
fn bce_with_logits_dispatches_with_wgpu_value_and_gradient_parity() {
    #[cfg(target_os = "windows")]
    std::env::set_var("WGPU_BACKEND", "dx12");
    if hephaestus_wgpu::WgpuDevice::try_default("coeus-wgpu-bce-with-logits-test").is_err() {
        assert_ne!(
            std::env::var("HEPHAESTUS_WGPU_REQUIRE_DEVICE").as_deref(),
            Ok("1"),
            "WGPU CI requires an acquired device"
        );
        return;
    }

    let logits = Tensor::from_slice_on([2, 2], &[100.0_f32, -100.0, 1.5, -0.5], &seq());
    let target = Tensor::from_slice_on([2, 2], &[1.0_f32, 0.0, 1.0, 0.0], &seq());
    let cpu_logits = Var::new(logits.clone().permute(&[1, 0]), true);
    let cpu_target = Var::new(target.clone().permute(&[1, 0]), true);
    let wgpu_logits = Var::new(to_gpu(&logits).permute(&[1, 0]), true);
    let wgpu_target = Var::new(to_gpu(&target).permute(&[1, 0]), true);

    let cpu_loss = coeus_nn::bce_with_logits(&cpu_logits, &cpu_target);
    let wgpu_loss = coeus_nn::bce_with_logits(&wgpu_logits, &wgpu_target);
    cpu_loss
        .backward()
        .expect("CPU BCE-with-logits backward must succeed");
    wgpu_loss
        .backward()
        .expect("WGPU BCE-with-logits backward must succeed");

    let wgpu_loss = to_cpu(&wgpu_loss.tensor);
    let cpu_logits_gradient = cpu_logits.grad().expect("tracked CPU logits gradient");
    let wgpu_logits_gradient = to_cpu(&wgpu_logits.grad().expect("tracked WGPU logits gradient"));
    let cpu_target_gradient = cpu_target.grad().expect("tracked CPU target gradient");
    let wgpu_target_gradient = to_cpu(&wgpu_target.grad().expect("tracked WGPU target gradient"));

    assert!(cpu_loss.tensor.as_slice()[0].is_finite());
    assert!(wgpu_loss.as_slice()[0].is_finite());

    assert_parity(
        "BCE-with-logits loss",
        cpu_loss.tensor.as_slice(),
        wgpu_loss.as_slice(),
    );
    assert_parity(
        "BCE-with-logits logits gradient",
        cpu_logits_gradient.as_slice(),
        wgpu_logits_gradient.as_slice(),
    );
    assert_parity(
        "BCE-with-logits target gradient",
        cpu_target_gradient.as_slice(),
        wgpu_target_gradient.as_slice(),
    );
}
